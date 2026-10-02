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

namespace sfp {

namespace {
double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
std::atomic<uint64_t> gFrameId{1};

std::shared_ptr<Frame> load(const std::string& path, int threads, std::string& error) {
    auto f = std::make_shared<Frame>();
    f->path = path;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> data;
    if (!read_file(path, data, error)) return nullptr;
    f->readMs = ms_since(t0);
    auto t1 = std::chrono::steady_clock::now();
    if (!parse_dng(data.data(), data.size(), f->info, error)) return nullptr;
    f->raw.resize(static_cast<size_t>(f->info.width) * f->info.height);
    if (!decode_raw(data.data(), data.size(), f->info, f->raw.data(), threads, error)) return nullptr;
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

std::shared_ptr<const Frame> FrameCache::fetch(const std::string& path, std::string& error) {
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

void FrameCache::prefetch(const std::vector<std::string>& paths) {
    {
        std::lock_guard<std::mutex> l(d->m);
        d->queue.assign(paths.begin(), paths.end());
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
