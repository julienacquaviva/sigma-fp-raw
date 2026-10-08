#include "frame_cache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "gyro.h"
#include "lens_profile.h"

namespace sfp {

namespace {
double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
std::atomic<uint64_t> gFrameId{1};

// Cache key of a frame with its shading map applied.
const char kShaded[] = "|shaded|";
std::string cache_key(const std::string& path, bool shading, const std::string& vignetteFile) { return shading ? path + kShaded + vignetteFile : path; }

// The part of the lens profile a clip's frames show. The camera writes one and the same profile
// (shading map and distortion terms) whatever the recording mode: bit for bit the same over
// 3:2, 16:9, 2:1 and 2.4:1 recordings and over every sensor window (38 clips, 2026-10-05 to
// 10-07). It is the profile of the whole 3:2 sensor; the frame's gyro block tells the window
// that was read, taken as centred. Kept per clip. Frames without a block: the whole profile.
void shading_window(const std::string& path, const DngInfo& info, double& fracW, double& fracH) {
    static std::mutex m;
    static std::unordered_map<std::string, std::pair<double, double>> known;
    std::string prefix, suffix;
    long long number = 0;
    int digits = 0;
    const std::string key = split_sequence(path, prefix, number, digits, suffix) ? prefix : path;
    {
        std::lock_guard<std::mutex> l(m);
        auto it = known.find(key);
        if (it != known.end()) { fracW = it->second.first; fracH = it->second.second; return; }
    }
    fracW = fracH = 1;
    GyroBlock b;
    if (read_fpg2(path, b)) {
        GyroHeader h;
        h.activeW = info.cropW; h.activeH = info.cropH;
        window_for(b.dcCrop, b.windowW, b.windowH, b.recordedW, b.recordedH, b.readoutUs, h);
        if (h.windowW > 0 && h.windowH > 0) {
            fracW = std::min(1.0, h.windowW / 6048.0);
            fracH = std::min(1.0, h.windowH / 4032.0);
        }
    }
    std::lock_guard<std::mutex> l(m);
    if (known.size() > 64) known.clear();
    known[key] = {fracW, fracH};
}

std::shared_ptr<Frame> load(const std::string& key, int threads, std::string& error) {
    auto f = std::make_shared<Frame>();
    const size_t mark = key.find(kShaded);
    const bool shading = mark != std::string::npos;
    const std::string path = shading ? key.substr(0, mark) : key;
    const std::string vignetteFile = shading ? key.substr(mark + sizeof kShaded - 1) : std::string();
    f->path = path;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> data;
    if (!read_file(path, data, error)) return nullptr;
    f->readMs = ms_since(t0);
    auto t1 = std::chrono::steady_clock::now();
    if (!parse_dng(data.data(), data.size(), f->info, error)) return nullptr;
    f->raw.resize(static_cast<size_t>(f->info.width) * f->info.height);
    if (!decode_raw(data.data(), data.size(), f->info, f->raw.data(), threads, error)) return nullptr;
    shading_window(path, f->info, f->profileW, f->profileH);
    if (shading) {
        // The vignette: the picked file, else the frame's own map, else the installed Adobe profile
        // (lens_vignette). Its brightness only, the same gain for the three colours: the colour
        // part of the camera's map (red and blue against green) is not applied. On frames of the
        // modified recording modes it puts green and magenta patches into a picture that has none
        // without it (seen 2026-10-08; the raw data evidently does not carry that colour shading).
        GainMap map;
        std::string note;
        bool ownColour = true;
        if (lens_vignette(vignetteFile, f->info, map, ownColour, note)) {
            GainMap brightness;
            brightness.rows = map.rows; brightness.cols = map.cols; brightness.planes = 1;
            brightness.areaW = map.areaW; brightness.areaH = map.areaH;
            brightness.gain.resize(static_cast<size_t>(map.rows) * map.cols);
            for (size_t i = 0; i < brightness.gain.size(); ++i) brightness.gain[i] = map.gain[i * map.planes + (map.planes == 3 ? 1 : 0)];
            apply_shading(f->info, brightness, f->raw.data(), f->profileW, f->profileH);
            f->shaded = true;
        }
    }
    f->decodeMs = ms_since(t1);
    f->id = gFrameId++;
    return f;
}
}  // namespace

struct FrameCache::Impl {
    struct Entry {
        std::shared_ptr<Frame> frame;
        bool loading = false;
        bool failed = false;
        std::string error;
        std::list<std::string>::iterator lru;
    };
    mutable std::mutex m;
    std::condition_variable cv, work;
    std::unordered_map<std::string, Entry> map;
    std::list<std::string> lru;   // front = most recent
    std::deque<std::string> queue;
    size_t bytes = 0, budget = size_t(2) << 30;
    bool stop = false;
    std::vector<std::thread> workers;
    Stats st{};

    void touch(Entry& e, const std::string& key) {
        lru.erase(e.lru);
        lru.push_front(key);
        e.lru = lru.begin();
    }
    void evict(size_t limit) {
        while (bytes > limit && lru.size() > 2) {
            auto it = std::prev(lru.end());
            auto e = map.find(*it);
            if (e != map.end() && e->second.loading) {   // never evict in-flight work
                lru.splice(lru.begin(), lru, it);
                break;
            }
            if (e != map.end()) {
                if (e->second.frame) bytes -= e->second.frame->raw.size() * 2;
                map.erase(e);
            }
            lru.erase(it);
        }
    }
    void insert_done(const std::string& key, std::shared_ptr<Frame> f, const std::string& err) {
        auto& e = map[key];
        e.loading = false;
        if (f) {
            e.frame = std::move(f);
            bytes += e.frame->raw.size() * 2;
        } else {
            e.failed = true;
            e.error = err;
        }
        evict(budget);
        cv.notify_all();
    }
    void run() {
        for (;;) {
            std::string key;
            {
                std::unique_lock<std::mutex> l(m);
                work.wait(l, [&] { return stop || !queue.empty(); });
                if (stop) return;
                key = queue.front();
                queue.pop_front();
                auto it = map.find(key);
                if (it != map.end()) continue;           // cached, loading or failed
                // Make room from the least recently used end (frames behind the playhead).
                evict(budget * 8 / 10);
                auto& e = map[key];
                e.loading = true;
                lru.push_front(key);
                e.lru = lru.begin();
            }
            std::string err;
            auto f = load(key, 1, err);
            std::lock_guard<std::mutex> l(m);
            if (f) st.prefetched++;
            else st.failed++;
            insert_done(key, std::move(f), err);
        }
    }
};

FrameCache& FrameCache::get() {
    static FrameCache c;
    return c;
}

FrameCache::FrameCache() : d(new Impl) {
    unsigned n = std::thread::hardware_concurrency();
    int workers = std::clamp<int>(static_cast<int>(n) / 3, 2, 8);
    for (int i = 0; i < workers; ++i) d->workers.emplace_back([this] { d->run(); });
}

FrameCache::~FrameCache() {
    {
        std::lock_guard<std::mutex> l(d->m);
        d->stop = true;
    }
    d->work.notify_all();
    for (auto& t : d->workers) t.join();
    delete d;
}

void FrameCache::set_budget(size_t b) {
    std::lock_guard<std::mutex> l(d->m);
    d->budget = std::max<size_t>(b, size_t(256) << 20);
    d->evict(d->budget);
}

std::shared_ptr<const Frame> FrameCache::fetch(const std::string& file, std::string& error, bool shading, const std::string& vignetteFile) {
    const std::string path = cache_key(file, shading, vignetteFile);
    std::unique_lock<std::mutex> l(d->m);
    auto it = d->map.find(path);
    if (it != d->map.end()) {
        if (it->second.loading) {
            d->st.waits++;
            d->cv.wait(l, [&] { auto j = d->map.find(path); return j == d->map.end() || !j->second.loading; });
            it = d->map.find(path);
        } else {
            d->st.hits++;
        }
        if (it != d->map.end()) {
            if (it->second.frame) { d->touch(it->second, path); return it->second.frame; }
            if (it->second.failed) {
                error = it->second.error;
                // Allow a retry later (file may appear or be repaired).
                d->lru.erase(it->second.lru);
                d->map.erase(it);
                return nullptr;
            }
        }
    }
    d->st.misses++;
    auto& e = d->map[path];
    e.loading = true;
    d->lru.push_front(path);
    e.lru = d->lru.begin();
    l.unlock();
    std::string err;
    int threads = std::clamp<int>(static_cast<int>(std::thread::hardware_concurrency()), 1, 16);
    auto f = load(path, threads, err);
    l.lock();
    std::shared_ptr<const Frame> result = f;
    d->insert_done(path, std::move(f), err);
    if (!result) {
        error = err;
        auto j = d->map.find(path);
        if (j != d->map.end()) { d->lru.erase(j->second.lru); d->map.erase(j); }
    }
    return result;
}

void FrameCache::prefetch(const std::vector<std::string>& paths, bool shading, const std::string& vignetteFile) {
    {
        std::lock_guard<std::mutex> l(d->m);
        d->queue.clear();
        for (const auto& p : paths) d->queue.push_back(cache_key(p, shading, vignetteFile));
    }
    d->work.notify_all();
}

FrameCache::Stats FrameCache::stats() const {
    std::lock_guard<std::mutex> l(d->m);
    Stats s = d->st;
    s.bytes = d->bytes;
    s.frames = d->map.size();
    return s;
}

bool split_sequence(const std::string& path, std::string& prefix, long long& number, int& digits, std::string& suffix) {
    size_t dot = path.find_last_of('.');
    size_t slash = path.find_last_of("/\\");
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) dot = path.size();
    size_t b = dot;
    while (b > 0 && b > (slash == std::string::npos ? 0 : slash + 1) && std::isdigit(static_cast<unsigned char>(path[b - 1]))) --b;
    if (b == dot || dot - b > 12) return false;
    prefix = path.substr(0, b);
    number = std::stoll(path.substr(b, dot - b));
    digits = static_cast<int>(dot - b);
    suffix = path.substr(dot);
    return true;
}

std::string sequence_path(const std::string& prefix, long long number, int digits, const std::string& suffix) {
    std::string n = std::to_string(number);
    if (static_cast<int>(n.size()) < digits) n.insert(0, digits - n.size(), '0');
    return prefix + n + suffix;
}

}  // namespace sfp
