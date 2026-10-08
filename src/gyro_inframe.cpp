// In-frame gyro data: an "FPG2" block in the last 3072 bytes of the Sigma MakerNote of every
// DNG frame (FPGYRO_INFRAME.md). This file reads the blocks, gathers them over a whole clip on
// a background thread, builds the same track the .FPG sidecar gives, and caches it.
#include "gyro.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <thread>

#include "dng.h"
#include "frame_cache.h"
#include "platform.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static std::wstring wide(const std::string& s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? n : 1, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    w.resize(n > 0 ? n - 1 : 0);
    return w;
}
#endif

namespace sfp {

namespace {

const long long kFastOffset = 0x10D56;   // block position with today's camera header
const size_t kBlockHeader = 48;
// The scan waits for the drive, not for the processor: on the camera SSD over USB one thread
// reads 70 frames a second, 8 threads 500, 32 threads 1300 (no gain beyond that).
const unsigned kScanThreads = 32;

uint16_t rd16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t rd32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24); }

using os::File;

// Little-endian TIFF: value/offset field of a tag in the IFD at `ifd` (reads the file).
bool find_tag(const File& f, uint32_t ifd, uint16_t tag, uint16_t& type, uint32_t& count, uint32_t& value) {
    uint8_t n2[2];
    if (f.read(ifd, n2, 2) != 2) return false;
    const unsigned n = rd16(n2);
    if (n == 0 || n > 1024) return false;
    std::vector<uint8_t> e(12 * n);
    if (f.read(static_cast<long long>(ifd) + 2, e.data(), e.size()) != e.size()) return false;
    for (unsigned i = 0; i < n; ++i) {
        const uint8_t* p = e.data() + 12 * i;
        if (rd16(p) == tag) { type = rd16(p + 2); count = rd32(p + 4); value = rd32(p + 8); return true; }
    }
    return false;
}

}  // namespace

double dng_exposure(const std::string& dngPath) {
    File f(dngPath);
    uint8_t head[8], rat[8];
    uint16_t type = 0;
    uint32_t count = 0, exif = 0, at = 0;
    if (!f.ok() || f.read(0, head, 8) != 8 || head[0] != 'I' || head[1] != 'I') return 0;
    if (!find_tag(f, rd32(head + 4), 0x8769, type, count, exif) || !find_tag(f, exif, 0x829A, type, count, at) || type != 5) return 0;
    if (f.read(at, rat, 8) != 8 || rd32(rat + 4) == 0) return 0;
    const double e = static_cast<double>(rd32(rat)) / rd32(rat + 4);
    return e > 0 && e < 10 ? e : 0;
}

// ---- one block ----

bool parse_fpg2(const uint8_t* d, size_t size, GyroBlock& out, std::string* why) {
    auto fail = [&](const char* text) { if (why) *why = text; out.present = false; return false; };
    out = GyroBlock{};
    if (size < kBlockHeader || std::memcmp(d, "FPG2", 4) != 0) return fail("no FPG2 block");
    if (rd16(d + 4) != 1) return fail("unsupported FPG2 version");
    const size_t header = rd16(d + 6);
    const size_t n = rd16(d + 16);
    if (header < kBlockHeader || header > size) return fail("bad FPG2 header size");
    if (n > static_cast<size_t>(kFpg2MaxSamples) || header + 6 * n > size) return fail("bad FPG2 sample count");
    out.frame = rd32(d + 8);
    out.clockUs = rd32(d + 12);
    out.ringIndex = rd16(d + 18);
    out.flags = rd16(d + 20);
    out.lost = rd16(d + 22);
    out.take = rd32(d + 24);
    out.readoutUs = rd32(d + 28);
    out.markDelayUs = static_cast<int32_t>(rd32(d + 32));
    out.sensorMode = rd16(d + 36);
    out.resolution = d[38]; out.dcCrop = d[39]; out.bitDepth = d[40];
    for (int i = 0; i < 3; ++i) out.axis[i] = static_cast<int8_t>(d[42 + i]);
    out.lsb = rd16(d + 46);
    if (size >= 3064) {
        out.windowW = rd16(d + 3056); out.windowH = rd16(d + 3058);
        out.recordedW = rd16(d + 3060); out.recordedH = rd16(d + 3062);
    }
    out.samples.resize(3 * n);
    for (size_t i = 0; i < 3 * n; ++i) out.samples[i] = static_cast<int16_t>(rd16(d + header + 2 * i));
    out.present = true;
    return true;
}

bool read_fpg2(const std::string& dngPath, GyroBlock& out, std::string* why) {
    out = GyroBlock{};
    File f(dngPath);
    if (!f.ok()) { if (why) *why = "cannot open the file"; return false; }
    uint8_t region[kFpg2Region];
    // Fast path: the fixed position of today's header.
    if (f.read(kFastOffset, region, sizeof region) == sizeof region && parse_fpg2(region, sizeof region, out, nullptr)) return true;
    // General path: IFD0 -> EXIF IFD (0x8769) -> MakerNote (0x927C); the block is its last 3072 bytes.
    uint8_t head[8];
    if (f.read(0, head, 8) != 8 || head[0] != 'I' || head[1] != 'I' || rd16(head + 2) != 42) { if (why) *why = "not a little-endian TIFF"; return false; }
    uint16_t type = 0;
    uint32_t count = 0, exif = 0, note = 0;
    if (!find_tag(f, rd32(head + 4), 0x8769, type, count, exif) || !find_tag(f, exif, 0x927C, type, count, note)) { if (why) *why = "no MakerNote"; return false; }
    if (count < 4096) { if (why) *why = "MakerNote too short"; return false; }
    const long long at = static_cast<long long>(note) + count - kFpg2Region;
    if (at == kFastOffset) { if (why) *why = "no FPG2 block"; return false; }
    if (f.read(at, region, sizeof region) != sizeof region) { if (why) *why = "cannot read the MakerNote"; return false; }
    return parse_fpg2(region, sizeof region, out, why);
}

// ---- track from the blocks of a clip ----

std::shared_ptr<GyroClip> GyroClip::from_blocks(const std::vector<GyroBlock>& blocks, const ClipGeometry& geo, std::string& e) {
    const int N = static_cast<int>(blocks.size());
    // One take only: a folder with frames of another take keeps the take most frames belong to.
    std::map<uint32_t, int> takes;
    for (const auto& b : blocks)
        if (b.present) ++takes[b.take];
    if (takes.empty()) { e = "no gyro blocks in the frames"; return nullptr; }
    uint32_t take = takes.begin()->first;
    for (const auto& t : takes)
        if (t.second > takes[take]) take = t.first;
    // Blocks in file order; the frame counter must rise with it.
    std::vector<int> kept;
    for (int i = 0; i < N; ++i) {
        const GyroBlock& b = blocks[i];
        if (!b.present || b.take != take) continue;
        if (!kept.empty() && b.frame <= blocks[kept.back()].frame) continue;
        kept.push_back(i);
    }
    const int K = static_cast<int>(kept.size());
    if (K < 2) { e = "fewer than two frames carry gyro data"; return nullptr; }
    if (geo.rasterW < 16 || geo.rasterH < 16) { e = "cannot read the DNG raster size"; return nullptr; }
    const GyroBlock& b0 = blocks[kept[0]];

    // Mark clock (wraps at 2^32 us) and samples per block.
    std::vector<double> clock(K, 0.0);
    std::vector<long long> lost(K, 0), gap(K, 0);
    uint64_t t = 0;
    std::vector<double> rates;
    for (int k = 1; k < K; ++k) {
        const GyroBlock& b = blocks[kept[k]];
        const uint32_t dt = b.clockUs - blocks[kept[k - 1]].clockUs;
        t += dt;
        clock[k] = static_cast<double>(t);
        lost[k] = (b.flags & 2) ? b.lost : 0;
        if (!lost[k] && dt > 0 && !b.samples.empty()) rates.push_back(b.samples.size() / 3 * 1e6 / dt);
    }
    if (clock[K - 1] <= 0) { e = "the gyro blocks carry no usable clock"; return nullptr; }
    double rate0 = 0;
    if (!rates.empty()) {
        std::nth_element(rates.begin(), rates.begin() + rates.size() / 2, rates.end());
        rate0 = rates[rates.size() / 2];
    }
    // The lost count says at most 99 (the camera's ring holds 600 samples, a block 500). After
    // a longer pause, and where blocks were damaged or removed, more samples are missing than
    // it says: the mark clock against the sample count gives the true length of the gap.
    std::vector<uint8_t> broken(K, 0);
    for (int k = 1; k < K; ++k) {
        gap[k] = lost[k];
        if (rate0 > 0) {
            const double expected = rate0 * (clock[k] - clock[k - 1]) * 1e-6;
            const long long have = lost[k] + static_cast<long long>(blocks[kept[k]].samples.size() / 3);
            const long long extra = std::llround(expected) - have;
            if (extra > 50 || (lost[k] >= 99 && extra > 0)) { gap[k] += extra; broken[k] = extra > 50; }
        }
    }
    long long total = 0;
    for (int k = 0; k < K; ++k) total += gap[k] + static_cast<long long>(blocks[kept[k]].samples.size() / 3);
    if (total < 2 || total > 400000000LL) { e = "bad gyro sample count"; return nullptr; }
    std::vector<int16_t> samples;
    samples.reserve(static_cast<size_t>(total) * 3);
    std::vector<long long> after(K, 0);
    long long lostTotal = 0;
    for (int k = 0; k < K; ++k) {
        const GyroBlock& b = blocks[kept[k]];
        if (gap[k] > 0) {
            // Lost samples: a straight line from the last sample before to the first after.
            int16_t from[3] = {0, 0, 0}, to[3];
            if (samples.size() >= 3) std::memcpy(from, &samples[samples.size() - 3], sizeof from);
            if (b.samples.size() >= 3) std::memcpy(to, b.samples.data(), sizeof to);
            else std::memcpy(to, from, sizeof to);
            for (long long j = 1; j <= gap[k]; ++j)
                for (int a = 0; a < 3; ++a)
                    samples.push_back(static_cast<int16_t>(std::lround(from[a] + (to[a] - from[a]) * static_cast<double>(j) / (gap[k] + 1))));
            lostTotal += gap[k];
        }
        samples.insert(samples.end(), b.samples.begin(), b.samples.end());
        after[k] = static_cast<long long>(samples.size() / 3);
    }
    GyroHeader h;
    h.version = 2;
    h.frameCount = static_cast<uint32_t>(N);
    h.sampleCount = static_cast<uint32_t>(samples.size() / 3);
    h.rateHz = (after[K - 1] - after[0]) * 1e6 / clock[K - 1];
    if (!(h.rateHz > 100 && h.rateHz < 100000)) { e = "the gyro blocks give an impossible sample rate"; return nullptr; }
    h.lsbPerDps = b0.lsb ? b0.lsb : 131;
    h.rasterW = geo.rasterW; h.rasterH = geo.rasterH;
    h.activeX = geo.activeX; h.activeY = geo.activeY; h.activeW = geo.activeW; h.activeH = geo.activeH;
    if (h.activeW <= 0 || h.activeH <= 0 || h.activeX < 0 || h.activeY < 0 || h.activeX + h.activeW > h.rasterW || h.activeY + h.activeH > h.rasterH) {
        h.activeX = h.activeY = 0; h.activeW = h.rasterW; h.activeH = h.rasterH;
    }
    h.fps = geo.fps > 1 ? geo.fps : (kept[K - 1] - kept[0]) * 1e6 / clock[K - 1];
    h.readoutS = b0.readoutUs * 1e-6;
    h.exposureS = geo.exposureS;
    h.focalMm = geo.focalMm;
    // Mark delay 0 = unknown: the camera's provisional rule (half the readout + 5 ms), or the HQ value.
    h.markDelayS = b0.markDelayUs ? b0.markDelayUs * 1e-6 : (h.readoutS > 0 ? h.readoutS / 2 + 0.005 : 0.0174);
    bool used[3] = {false, false, false}, axesOk = true;
    for (int i = 0; i < 3; ++i) {
        int a = std::abs(b0.axis[i]);
        if (a < 1 || a > 3 || used[a - 1]) { axesOk = false; break; }
        used[a - 1] = true;
        h.axis[i] = b0.axis[i];
    }
    if (!axesOk) { e = "bad axis map in the gyro blocks"; return nullptr; }
    h.flags = 0;
    h.preroll = static_cast<uint32_t>(after[0]);
    h.pitchNm = 0;
    h.resolution = b0.resolution; h.dcCrop = b0.dcCrop; h.bitDepth = b0.bitDepth; h.sensorMode = b0.sensorMode;
    window_for(b0.dcCrop, b0.windowW, b0.windowH, b0.recordedW, b0.recordedH, b0.readoutUs, h);

    // Frame table: the count at each frame's mark. A frame without a block between two blocks
    // whose samples are all there (they are in the next block's slice) gets its mark by
    // interpolation; with samples missing, the frames in between have no gyro data.
    std::vector<uint32_t> table(N, 0);
    std::vector<uint8_t> valid(N, 0);
    const double perFrame = h.fps > 1 ? h.rateHz / h.fps : 0;
    for (int i = 0; i < kept[0]; ++i) table[i] = static_cast<uint32_t>(std::max(0.0, after[0] - perFrame * (kept[0] - i)));
    for (int k = 0; k < K; ++k) {
        table[kept[k]] = static_cast<uint32_t>(after[k]);
        valid[kept[k]] = 1;
        if (k == 0) continue;
        const int a = kept[k - 1], b = kept[k];
        for (int i = a + 1; i < b; ++i) {
            table[i] = static_cast<uint32_t>(after[k - 1] + std::llround(static_cast<double>(after[k] - after[k - 1]) * (i - a) / (b - a)));
            // Samples missing here: the frame is not stabilised. A number without a file (a frame
            // that was never written, or was removed) is not shown, so it does not count.
            valid[i] = (gap[k] == 0 && !broken[k]) || !blocks[i].fileExists;
        }
    }
    for (int i = kept[K - 1] + 1; i < N; ++i) table[i] = static_cast<uint32_t>(after[K - 1]);
    auto c = assemble(h, std::move(table), std::move(valid), samples.data(), e);
    if (!c) return nullptr;
    c->inFrame = true;
    c->name = "in-frame";
    c->blocks = K;
    c->lostSamples = lostTotal;
    c->raw_ = std::move(samples);
    return c;
}

// ---- clip scan, cache file, background store ----

namespace {

struct SeqFile { long long number; uint64_t time, size; std::string path; };

std::vector<SeqFile> list_sequence(const std::string& dngPath, std::string& key) {
    std::vector<SeqFile> out;
    std::string prefix, suffix;
    long long number = 0;
    int digits = 0;
    if (!split_sequence(dngPath, prefix, number, digits, suffix)) {
        key = dngPath;
        os::FileInfo a;
        if (os::stat_file(dngPath, a)) out.push_back({1, a.time, a.size, dngPath});
        return out;
    }
    key = prefix;
    const size_t slash = prefix.find_last_of("/\\");
    const std::string dir = slash == std::string::npos ? std::string() : prefix.substr(0, slash + 1);
    const size_t stemLen = prefix.size() - dir.size();
    for (const auto& f : os::list_matching(prefix, suffix)) {
        const std::string& name = f.name;
        if (name.size() <= stemLen + suffix.size()) continue;
        const std::string num = name.substr(stemLen, name.size() - stemLen - suffix.size());
        if (num.empty() || num.size() > 12 || !std::all_of(num.begin(), num.end(), [](char ch) { return ch >= '0' && ch <= '9'; })) continue;
        const long long n = std::stoll(num);
        if (n < 1 || n > 100000000) continue;
        out.push_back({n, f.time, f.size, dir + name});
    }
    std::sort(out.begin(), out.end(), [](const SeqFile& x, const SeqFile& y) { return x.number < y.number; });
    return out;
}

// Raster, active area, frame rate and focal length of the clip, from one frame's header.
bool clip_geometry(const std::string& path, ClipGeometry& geo, std::string& err) {
    std::vector<uint8_t> data;
    {
        File f(path);
        if (!f.ok()) { err = "cannot open it"; return false; }
        data.resize(512 * 1024);
        data.resize(f.read(0, data.data(), data.size()));
    }
    // Only the tags are needed, and they are in the head of the file. The DNG parser checks
    // that the image data lies inside its buffer, so the buffer is extended with zeros
    // instead of reading the image (also lets a header-only test frame through).
    const size_t head = data.size();
    DngInfo info;
    data.resize(std::max<size_t>(head, 128u << 20));
    const bool parsed = parse_dng(data.data(), data.size(), info, err);
    data.resize(head);
    if (!parsed) return false;
    geo.rasterW = info.width; geo.rasterH = info.height;
    geo.activeX = info.activeLeft + info.cropX; geo.activeY = info.activeTop + info.cropY;
    geo.activeW = info.cropW; geo.activeH = info.cropH;
    geo.fps = info.fps;
    // EXIF FocalLength (0x920A) and ExposureTime (0x829A), rationals.
    auto u16at = [&](size_t o) { return o + 2 <= data.size() ? rd16(&data[o]) : 0; };
    auto u32at = [&](size_t o) { return o + 4 <= data.size() ? rd32(&data[o]) : 0u; };
    auto tag = [&](size_t ifd, uint16_t want, uint32_t& value) {
        const unsigned n = u16at(ifd);
        for (unsigned i = 0; i < n && i < 1024; ++i)
            if (u16at(ifd + 2 + 12 * i) == want) { value = u32at(ifd + 2 + 12 * i + 8); return true; }
        return false;
    };
    auto rational = [&](size_t o) {
        const double num = u32at(o), den = u32at(o + 4);
        return den > 0 ? num / den : 0.0;
    };
    uint32_t exif = 0, v = 0;
    if (data.size() > 8 && data[0] == 'I' && tag(u32at(4), 0x8769, exif)) {
        if (tag(exif, 0x920A, v)) geo.focalMm = rational(v);
        if (tag(exif, 0x829A, v)) geo.exposureS = rational(v);
    }
    return info.width >= 16 && info.height >= 16;
}

// ---- cache file ----

struct Signature { uint64_t count = 0, firstTime = 0, lastTime = 0, firstSize = 0, lastSize = 0; };

Signature signature(const std::vector<SeqFile>& files) {
    Signature s;
    if (files.empty()) return s;
    s.count = files.size();
    s.firstTime = files.front().time; s.firstSize = files.front().size;
    s.lastTime = files.back().time; s.lastSize = files.back().size;
    return s;
}

std::string cache_dir() {
    std::string dir = os::env("SFP_GYRO_CACHE_DIR");
    if (dir == "off" || dir == "0") return "";
    if (dir.empty()) {
        std::string base = os::user_cache_dir();
        if (base.empty()) return "";
        os::make_dir(base + os::kSep + "SigmaFpRaw");
        dir = base + os::kSep + "SigmaFpRaw" + os::kSep + "gyro";
    }
    os::make_dir(dir);
    return dir;
}

std::string cache_path(const std::string& key) {
    const std::string dir = cache_dir();
    if (dir.empty()) return "";
    uint64_t hsh = 1469598103934665603ull;
    for (char ch : key) {
        hsh ^= static_cast<uint8_t>(std::toupper(static_cast<unsigned char>(ch == '/' ? '\\' : ch)));
        hsh *= 1099511628211ull;
    }
    char name[40];
    std::snprintf(name, sizeof name, "%016llx.fpgc", static_cast<unsigned long long>(hsh));
    return dir + os::kSep + name;
}

const uint32_t kCacheVersion = 3;

template <class T> void put(std::vector<uint8_t>& o, const T& v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    o.insert(o.end(), p, p + sizeof v);
}

void save_cache(const std::string& key, const Signature& sig, const GyroClip* clip) {
    const std::string path = cache_path(key);
    if (path.empty()) return;
    std::vector<uint8_t> o;
    o.insert(o.end(), {'F', 'P', 'G', 'C'});
    put(o, kCacheVersion);
    put(o, static_cast<uint32_t>(sizeof(GyroHeader)));
    put(o, sig);
    put(o, static_cast<uint32_t>(key.size()));
    o.insert(o.end(), key.begin(), key.end());
    put(o, static_cast<uint32_t>(clip ? 1 : 0));
    if (clip) {
        put(o, clip->h);
        put(o, static_cast<int32_t>(clip->blocks));
        put(o, static_cast<int64_t>(clip->lostSamples));
        const auto& table = clip->table();
        const auto& valid = clip->valid();
        const auto& raw = clip->raw();
        put(o, static_cast<uint32_t>(clip->exposureByFrame.size()));
        for (float v : clip->exposureByFrame) put(o, v);
        const size_t at = o.size();
        o.resize(at + table.size() * 4 + valid.size() + raw.size() * 2);
        std::memcpy(&o[at], table.data(), table.size() * 4);
        std::memcpy(&o[at + table.size() * 4], valid.data(), valid.size());
        std::memcpy(&o[at + table.size() * 4 + valid.size()], raw.data(), raw.size() * 2);
    }
    const std::string tmp = path + ".tmp";
    if (!os::write_file(tmp, o.data(), o.size()) || !os::replace_file(tmp, path)) { os::remove_file(tmp); return; }
    // Keep the folder small: drop the oldest files beyond 1 GiB.
    const std::string dir = cache_dir();
    std::vector<os::FileInfo> all = os::list_matching(dir + os::kSep, ".fpgc");
    uint64_t sum = 0;
    for (const auto& x : all) sum += x.size;
    std::sort(all.begin(), all.end(), [](const os::FileInfo& x, const os::FileInfo& y) { return x.time < y.time; });
    for (size_t i = 0; i + 1 < all.size() && sum > (1ull << 30); ++i) {
        if (os::remove_file(dir + os::kSep + all[i].name)) sum -= all[i].size;
    }
}

// True when a cache file for exactly this clip state exists; clip stays null for "no gyro data".
bool load_cache(const std::string& key, const Signature& sig, std::shared_ptr<GyroClip>& clip) {
    const std::string path = cache_path(key);
    if (path.empty()) return false;
    std::vector<uint8_t> d;
    std::string err;
    if (!os::exists(path) || !read_file(path, d, err)) return false;
    size_t at = 0;
    auto get = [&](void* out, size_t n) {
        if (at + n > d.size()) return false;
        std::memcpy(out, &d[at], n);
        at += n;
        return true;
    };
    char magic[4];
    uint32_t version = 0, hsize = 0, klen = 0, has = 0;
    Signature s;
    if (!get(magic, 4) || std::memcmp(magic, "FPGC", 4) != 0 || !get(&version, 4) || version != kCacheVersion || !get(&hsize, 4) ||
        hsize != sizeof(GyroHeader) || !get(&s, sizeof s) || std::memcmp(&s, &sig, sizeof s) != 0 || !get(&klen, 4) || at + klen > d.size() ||
        std::string(reinterpret_cast<const char*>(&d[at]), klen) != key) return false;
    at += klen;
    if (!get(&has, 4)) return false;
    if (!has) { clip = nullptr; return true; }
    GyroHeader h;
    int32_t blocks = 0;
    int64_t lost = 0;
    if (!get(&h, sizeof h) || !get(&blocks, 4) || !get(&lost, 8)) return false;
    uint32_t ne = 0;
    if (!get(&ne, 4) || ne > 100000000u) return false;
    std::vector<float> exposure(ne);
    if (ne && !get(exposure.data(), ne * sizeof(float))) return false;
    const size_t n = h.frameCount, m = static_cast<size_t>(h.sampleCount) * 3;
    if (h.frameCount != sig.count && h.frameCount < 1) return false;
    if (at + n * 4 + n + m * 2 != d.size()) return false;
    std::vector<uint32_t> table(n);
    std::vector<uint8_t> valid(n);
    std::vector<int16_t> raw(m);
    get(table.data(), n * 4);
    get(valid.data(), n);
    get(raw.data(), m * 2);
    auto c = GyroClip::assemble(h, std::move(table), std::move(valid), raw.data(), err);
    if (!c) return false;
    c->inFrame = true;
    c->name = "in-frame";
    c->blocks = blocks;
    c->lostSamples = lost;
    c->exposureByFrame = std::move(exposure);
    clip = c;
    return true;
}

// ---- background store ----

struct Entry {
    std::atomic<bool> finished{false};
    std::atomic<long long> done{0}, total{0};
    std::shared_ptr<const GyroClip> clip;     // set before finished
    bool fromCache = false;
    double ms = 0;
    std::string note;                         // why blocks were found but no track came of them
    std::chrono::steady_clock::time_point at;
};

std::mutex gStoreMutex;
std::map<std::string, std::shared_ptr<Entry>> gStore;
std::atomic<int> gWorkers{0};
std::atomic<bool> gCancel{false};

void scan(std::shared_ptr<Entry> e, std::string dngPath) {
    const auto t0 = std::chrono::steady_clock::now();
    std::string key;
    std::vector<SeqFile> files = list_sequence(dngPath, key);
    std::shared_ptr<GyroClip> clip;
    if (!files.empty()) {
        const Signature sig = signature(files);
        const long long frames = files.back().number;
        e->total = static_cast<long long>(files.size());
        if (load_cache(key, sig, clip)) {
            e->fromCache = true;
        } else {
            const long long delayUs = std::atoll(os::env("SFP_GYRO_SCAN_DELAY_US").c_str());   // test hook: slow media
            // A quick look first: clips without any block (older takes) are not read frame by frame.
            bool any = false;
            const size_t step = std::max<size_t>(1, files.size() / 48);
            GyroBlock probe;
            if (read_fpg2(dngPath, probe)) any = true;
            for (size_t i = 0; i < files.size() && !any && !gCancel; i += step) any = read_fpg2(files[i].path, probe);
            if (!any && !gCancel) any = read_fpg2(files.back().path, probe);
            if (any && frames <= 20000000) {
                std::vector<GyroBlock> blocks(static_cast<size_t>(frames));
                std::atomic<size_t> next{0};
                const unsigned threads = kScanThreads;
                auto work = [&] {
                    for (;;) {
                        const size_t i = next.fetch_add(1);
                        if (i >= files.size() || gCancel) return;
                        GyroBlock& blk = blocks[static_cast<size_t>(files[i].number - 1)];
                        read_fpg2(files[i].path, blk);
                        blk.fileExists = true;
                        if (delayUs > 0) std::this_thread::sleep_for(std::chrono::microseconds(delayUs));
                        ++e->done;
                    }
                };
                std::vector<std::thread> pool;
                for (unsigned i = 1; i < threads; ++i) pool.emplace_back(work);
                work();
                for (auto& th : pool) th.join();
                ClipGeometry geo;
                std::string err;
                if (!gCancel) {
                    if (!clip_geometry(files.front().path, geo, err)) err = "cannot read the first DNG (" + err + ")";
                    else clip = GyroClip::from_blocks(blocks, geo, err);
                    if (!clip) e->note = err;
                    if (clip) {
                        // Exposure time: the same in every frame, or (auto exposure) followed over the clip.
                        const size_t picks = std::min<size_t>(files.size(), 16);
                        std::vector<std::pair<long long, double>> seen;
                        for (size_t k = 0; k < picks; ++k) {
                            const SeqFile& sf = files[picks > 1 ? k * (files.size() - 1) / (picks - 1) : 0];
                            const double ex = dng_exposure(sf.path);
                            if (ex > 0) seen.push_back({sf.number - 1, ex});
                        }
                        double lo = 1e9, hi = 0;
                        for (const auto& v : seen) { lo = std::min(lo, v.second); hi = std::max(hi, v.second); }
                        if (seen.size() >= 2 && hi > lo * 1.02) {
                            clip->exposureByFrame.resize(static_cast<size_t>(frames));
                            size_t k = 0;
                            for (long long i = 0; i < frames; ++i) {
                                while (k + 1 < seen.size() && seen[k + 1].first <= i) ++k;
                                double v = seen[k].second;
                                if (k + 1 < seen.size() && i > seen[k].first)
                                    v += (seen[k + 1].second - v) * static_cast<double>(i - seen[k].first) / (seen[k + 1].first - seen[k].first);
                                clip->exposureByFrame[static_cast<size_t>(i)] = static_cast<float>(v);
                            }
                        }
                    }
                }
            }
            if (!gCancel && (clip || e->note.empty())) save_cache(key, sig, clip.get());
            if (clip) clip->drop_raw();
        }
    }
    e->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    e->clip = clip;
    e->at = std::chrono::steady_clock::now();
    e->finished = true;
    --gWorkers;
}

}  // namespace

bool dng_geometry(const std::string& dngPath, ClipGeometry& geo) {
    std::string err;
    return clip_geometry(dngPath, geo, err);
}

GyroLoad inframe_gyro(const std::string& dngPath) {
    std::string prefix, suffix, key = dngPath;
    long long number = 0;
    int digits = 0;
    if (split_sequence(dngPath, prefix, number, digits, suffix)) key = prefix;
    std::shared_ptr<Entry> e;
    {
        std::lock_guard<std::mutex> l(gStoreMutex);
        auto it = gStore.find(key);
        // A clip without gyro data is looked at again after a while (files may have been added).
        if (it != gStore.end() && it->second->finished && !it->second->clip &&
            std::chrono::steady_clock::now() - it->second->at > std::chrono::seconds(20)) {
            gStore.erase(it);
            it = gStore.end();
        }
        if (it != gStore.end()) e = it->second;
    }
    if (!e) {
        // New clip. With a block in this frame, or no sidecar to use instead, gather the clip in
        // the background; a clip of the sidecar kind (no block here, an .FPG next to it) is
        // answered at once so that it never shows an unstabilised frame.
        GyroBlock probe;
        std::string why;
        const bool here = read_fpg2(dngPath, probe);
        auto fresh = std::make_shared<Entry>();
        if (!here && !find_fpg(dngPath, why).empty()) {
            fresh->at = std::chrono::steady_clock::now();
            fresh->finished = true;
        }
        std::lock_guard<std::mutex> l(gStoreMutex);
        auto it = gStore.find(key);
        if (it != gStore.end()) {
            e = it->second;
        } else {
            // Keep the tracks in memory within about 20 million samples (2 hours of footage,
            // 32 bytes each): drop the finished ones when a new clip comes in beyond that.
            uint64_t held = 0;
            for (const auto& i : gStore)
                if (i.second->finished && i.second->clip) held += i.second->clip->h.sampleCount;
            if (gStore.size() > 64 || held > 20000000ull) {
                for (auto i = gStore.begin(); i != gStore.end();) i = i->second->finished ? gStore.erase(i) : std::next(i);
            }
            e = gStore[key] = fresh;
            if (!e->finished) {
                ++gWorkers;
                std::thread(scan, e, dngPath).detach();
            }
        }
        // Short clips and cache hits are ready within moments: give them that long.
        const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(120);
        while (!e->finished && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    GyroLoad out;
    out.frames = e->total;
    if (!e->finished) {
        out.loading = true;
        const long long total = e->total;
        out.percent = total > 0 ? static_cast<int>(std::min<long long>(99, e->done * 100 / total)) : 0;
        return out;
    }
    out.clip = e->clip;
    out.fromCache = e->fromCache;
    out.scanMs = e->ms;
    out.note = e->note;
    return out;
}

GyroLoad inframe_gyro_wait(const std::string& dngPath) {
    for (;;) {
        GyroLoad l = inframe_gyro(dngPath);
        if (!l.loading) return l;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

double scan_blocks(const std::string& dngPath, long long& count, long long& found, int threads, bool fixedOffsetOnly, bool unbuffered) {
    const auto t0 = std::chrono::steady_clock::now();
    std::string key;
    std::vector<SeqFile> files = list_sequence(dngPath, key);
    std::atomic<size_t> next{0};
    std::atomic<long long> hits{0};
    auto work = [&] {
        GyroBlock b;
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= files.size()) return;
            if (fixedOffsetOnly) {
                // What a frame with a block costs: one open and one read at the fixed position.
#ifdef _WIN32
                if (unbuffered) {
                    // Past the system file cache: what the drive itself needs (timing only).
                    HANDLE h = CreateFileW(wide(files[i].path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
                    if (h == INVALID_HANDLE_VALUE) continue;
                    static thread_local void* buf = VirtualAlloc(nullptr, 8192, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
                    LARGE_INTEGER at;
                    at.QuadPart = kFastOffset & ~4095LL;
                    DWORD got = 0;
                    const size_t skip = static_cast<size_t>(kFastOffset - at.QuadPart);
                    if (SetFilePointerEx(h, at, nullptr, FILE_BEGIN) && ReadFile(h, buf, 8192, &got, nullptr) && got >= skip + kFpg2Region &&
                        parse_fpg2(static_cast<uint8_t*>(buf) + skip, kFpg2Region, b, nullptr)) ++hits;
                    CloseHandle(h);
                    continue;
                }
#else
                (void)unbuffered;
#endif
                File f(files[i].path);
                uint8_t region[kFpg2Region];
                if (f.ok() && f.read(kFastOffset, region, sizeof region) == sizeof region && parse_fpg2(region, sizeof region, b, nullptr)) ++hits;
            } else if (read_fpg2(files[i].path, b)) ++hits;
        }
    };
    std::vector<std::thread> pool;
    const unsigned n = threads > 0 ? static_cast<unsigned>(threads) : kScanThreads;
    for (unsigned i = 1; i < n; ++i) pool.emplace_back(work);
    work();
    for (auto& th : pool) th.join();
    count = static_cast<long long>(files.size());
    found = hits;
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int frame_binning(const std::string& dngPath, int rasterW, int rasterH, std::string& why) {
    if (!(rasterW == 3024 && rasterH == 2010)) { why = "not a 3K frame"; return 0; }
    std::string prefix, suffix, key = dngPath;
    long long number = 0;
    int digits = 0;
    if (split_sequence(dngPath, prefix, number, digits, suffix)) key = prefix;
    static std::mutex m;
    static std::map<std::string, std::pair<int, std::string>> seen;
    {
        std::lock_guard<std::mutex> l(m);
        auto it = seen.find(key);
        if (it != seen.end()) { why = it->second.second; return it->second.first; }
    }
    GyroBlock b;
    int r;
    if (!read_fpg2(dngPath, b)) { r = -1; why = "3K frame, no gyro block: taken as the 2x2 binned readout"; }
    else if (b.windowW > 0) { r = 0; why = "1:1 sensor window (R124)"; }
    else if (b.dcCrop == 2) { r = 0; why = "1:1 sensor window (MQ 50, crop code 2)"; }
    else if (b.readoutUs == 18926) { r = 0; why = "1:1 sensor window (MQ 50, R122 readout)"; }
    else { r = 1; why = "2x2 binned readout (M98)"; }
    std::lock_guard<std::mutex> l(m);
    if (seen.size() > 256) seen.clear();
    seen[key] = {r, why};
    return r;
}

void gyro_shutdown() {
    gCancel = true;
    for (int i = 0; i < 5000 && gWorkers > 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    gCancel = false;
    std::lock_guard<std::mutex> l(gStoreMutex);
    for (auto i = gStore.begin(); i != gStore.end();) i = i->second->finished ? gStore.erase(i) : std::next(i);
}

}  // namespace sfp
