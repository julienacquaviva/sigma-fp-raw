#include "dng.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <cstring>
#include <thread>
#include <vector>

#include "lj92.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace sfp {
namespace {

struct Reader {
    const uint8_t* d;
    size_t n;
    bool le;
    bool ok(size_t o, size_t k) const { return o <= n && k <= n - o; }
    uint16_t u16(size_t o) const { return le ? d[o] | d[o + 1] << 8 : d[o] << 8 | d[o + 1]; }
    uint32_t u32(size_t o) const {
        return le ? d[o] | d[o + 1] << 8 | d[o + 2] << 16 | static_cast<uint32_t>(d[o + 3]) << 24
                  : static_cast<uint32_t>(d[o]) << 24 | d[o + 1] << 16 | d[o + 2] << 8 | d[o + 3];
    }
};

const int kTypeSize[] = {0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 4};

struct Entry {
    uint16_t tag, type;
    uint32_t count;
    size_t at;   // value offset
};

double value(const Reader& r, const Entry& e, uint32_t i) {
    size_t o = e.at + static_cast<size_t>(i) * kTypeSize[e.type < 14 ? e.type : 0];
    switch (e.type) {
        case 1: case 7: return r.d[o];
        case 6: return static_cast<int8_t>(r.d[o]);
        case 3: return r.u16(o);
        case 8: return static_cast<int16_t>(r.u16(o));
        case 4: case 13: return r.u32(o);
        case 9: return static_cast<int32_t>(r.u32(o));
        case 5: { double den = r.u32(o + 4); return den ? r.u32(o) / den : 0; }
        case 10: { double den = static_cast<int32_t>(r.u32(o + 4)); return den ? static_cast<int32_t>(r.u32(o)) / den : 0; }
        case 11: { float f; uint32_t v = r.u32(o); std::memcpy(&f, &v, 4); return f; }
        default: return 0;
    }
}

bool read_ifd(const Reader& r, size_t off, std::vector<Entry>& out) {
    if (!r.ok(off, 2)) return false;
    int count = r.u16(off);
    if (!r.ok(off + 2, static_cast<size_t>(count) * 12)) return false;
    for (int i = 0; i < count; ++i) {
        size_t e = off + 2 + 12 * i;
        Entry x{r.u16(e), r.u16(e + 2), r.u32(e + 4), 0};
        if (x.type == 0 || x.type > 13) continue;
        size_t bytes = static_cast<size_t>(x.count) * kTypeSize[x.type];
        x.at = bytes <= 4 ? e + 8 : r.u32(e + 8);
        if (!r.ok(x.at, bytes)) continue;
        out.push_back(x);
    }
    return true;
}

}  // namespace

void apply_shading(DngInfo& info, const GainMap& g, uint16_t* raw, double fracW, double fracH) {
    if (!g.valid()) return;
    // The result is kept at a finer scale than the camera's raw steps: all samples, black and
    // white levels times `fine` (16 for 12-bit data). Rounded back to whole raw steps, a dim
    // surface of 10 to 40 steps would be off by up to a few percent, differently for each
    // colour and in bands that follow the gain: green and magenta patches (seen 2026-10-08).
    int fine = 1;
    while (fine < 16 && info.white * (fine * 2) <= 65535.f) fine *= 2;
    const int W = info.width, H = info.height;
    const int ax = info.activeLeft + info.cropX, ay = info.activeTop + info.cropY;
    fracW = std::clamp(fracW, 0.05, 1.0);
    fracH = std::clamp(fracH, 0.05, 1.0);
    // Map column of every raster column (the map's grid spans the picture, edge to edge).
    std::vector<int> j0(W);
    std::vector<float> fb(W);
    for (int x = 0; x < W; ++x) {
        const double u = std::clamp((x - ax + 0.5) / info.cropW, 0.0, 1.0);
        const double m = (0.5 + (u - 0.5) * fracW) * (g.cols - 1);
        j0[x] = std::min(static_cast<int>(m), g.cols - 2);
        fb[x] = static_cast<float>(m - j0[x]);
    }
    const int white = static_cast<int>(info.white);
    const float top = info.white * fine;
    std::vector<float> row[2];
    for (int y = 0; y < H; ++y) {
        const double v = std::clamp((y - ay + 0.5) / info.cropH, 0.0, 1.0);
        const double m = (0.5 + (v - 0.5) * fracH) * (g.rows - 1);
        const int i0 = std::min(static_cast<int>(m), g.rows - 2);
        const float fa = static_cast<float>(m - i0);
        // The map's row for the two colours of this raster row.
        float black[2];
        for (int par = 0; par < 2; ++par) {
            const int k = ((y & 1) << 1) | par;
            const int plane = g.planes == 3 ? info.cfa[k] : 0;
            black[par] = info.black[k];
            row[par].resize(g.cols);
            const float* a = &g.gain[(static_cast<size_t>(i0) * g.cols) * g.planes + plane];
            const float* b = a + static_cast<size_t>(g.cols) * g.planes;
            for (int j = 0; j < g.cols; ++j) row[par][j] = a[j * g.planes] + fa * (b[j * g.planes] - a[j * g.planes]);
        }
        uint16_t* px = raw + static_cast<size_t>(y) * W;
        for (int x = 0; x < W; ++x) {
            const int c = px[x];
            if (c >= white) { px[x] = static_cast<uint16_t>(top); continue; }   // clipped stays clipped
            const int par = x & 1;
            const float* r = row[par].data() + j0[x];
            const float gain = r[0] + fb[x] * (r[1] - r[0]);
            const float out = (black[par] + (c - black[par]) * gain) * fine + 0.5f;
            px[x] = static_cast<uint16_t>(out < 0.f ? 0.f : out > top ? top : out);
        }
    }
    for (int k = 0; k < 4; ++k) info.black[k] *= fine;
    info.white *= fine;
}

bool read_file(const std::string& path, std::vector<uint8_t>& data, std::string& error) {
#ifdef _WIN32
    int wn = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring w(wn, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &w[0], wn);
    HANDLE h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE) { error = "Cannot open " + path; return false; }
    LARGE_INTEGER size;
    if (!GetFileSizeEx(h, &size) || size.QuadPart > (1LL << 31)) { CloseHandle(h); error = "Bad file size: " + path; return false; }
    data.resize(static_cast<size_t>(size.QuadPart));
    size_t done = 0;
    while (done < data.size()) {
        DWORD got = 0;
        DWORD want = static_cast<DWORD>(std::min<size_t>(data.size() - done, 64u << 20));
        if (!ReadFile(h, data.data() + done, want, &got, nullptr) || !got) break;
        done += got;
    }
    CloseHandle(h);
    if (done != data.size()) { error = "Read failed: " + path; return false; }
    return true;
#else
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { error = "Cannot open " + path; return false; }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    data.resize(n);
    bool ok = std::fread(data.data(), 1, n, f) == static_cast<size_t>(n);
    std::fclose(f);
    if (!ok) error = "Read failed: " + path;
    return ok;
#endif
}

bool parse_dng(const uint8_t* data, size_t size, DngInfo& info, std::string& error) {
    if (size < 16 || !((data[0] == 'I' && data[1] == 'I') || (data[0] == 'M' && data[1] == 'M'))) {
        error = "Not a TIFF/DNG file";
        return false;
    }
    Reader r{data, size, data[0] == 'I'};
    if (r.u16(2) != 42) { error = "Not a TIFF/DNG file"; return false; }
    std::vector<Entry> ifd0;
    if (!read_ifd(r, r.u32(4), ifd0)) { error = "Bad IFD0"; return false; }
    // CinemaDNG from the fp keeps the raw image in IFD0; also accept a raw SubIFD
    // (NewSubFileType 0) for stills-style files.
    std::vector<Entry> raw = ifd0, exif;
    auto find = [](const std::vector<Entry>& v, uint16_t tag) -> const Entry* {
        for (auto& e : v) if (e.tag == tag) return &e;
        return nullptr;
    };
    if (auto s = find(ifd0, 0x14A)) {
        for (uint32_t i = 0; i < s->count; ++i) {
            std::vector<Entry> sub;
            if (read_ifd(r, static_cast<size_t>(value(r, *s, i)), sub)) {
                auto t = find(sub, 0xFE);
                auto ph = find(sub, 0x106);
                if ((!t || value(r, *t, 0) == 0) && ph && value(r, *ph, 0) == 32803) { raw = sub; break; }
            }
        }
    }
    if (auto e = find(ifd0, 0x8769)) read_ifd(r, static_cast<size_t>(value(r, *e, 0)), exif);
    auto num = [&](const std::vector<Entry>& v, uint16_t tag, double def, uint32_t i = 0) {
        auto e = find(v, tag);
        return e && i < e->count ? value(r, *e, i) : def;
    };
    auto any = [&](uint16_t tag, double def, uint32_t i = 0) {
        return find(raw, tag) ? num(raw, tag, def, i) : num(ifd0, tag, def, i);
    };
    info = DngInfo{};
    info.width = static_cast<int>(num(raw, 0x100, 0));
    info.height = static_cast<int>(num(raw, 0x101, 0));
    info.bps = static_cast<int>(num(raw, 0x102, 0));
    info.compression = static_cast<int>(num(raw, 0x103, 1));
    if (num(raw, 0x106, 0) != 32803) { error = "Not a CFA DNG"; return false; }
    if (info.width <= 0 || info.height <= 0 || info.width > 16384 || info.height > 16384 ||
        info.bps < 8 || info.bps > 16) { error = "Unsupported raw dimensions"; return false; }
    const Entry* offs = find(raw, 0x144);
    const Entry* cnts = find(raw, 0x145);
    if (offs && cnts) {
        info.tileW = static_cast<int>(num(raw, 0x142, 0));
        info.tileH = static_cast<int>(num(raw, 0x143, 0));
    } else {
        offs = find(raw, 0x111);
        cnts = find(raw, 0x117);
    }
    if (!offs || !cnts || offs->count != cnts->count || !offs->count) { error = "Missing raw data offsets"; return false; }
    for (uint32_t i = 0; i < offs->count; ++i) {
        info.offsets.push_back(static_cast<uint64_t>(value(r, *offs, i)));
        info.counts.push_back(static_cast<uint64_t>(value(r, *cnts, i)));
        if (!r.ok(info.offsets.back(), info.counts.back())) { error = "Raw data outside file"; return false; }
    }
    if (auto p = find(raw, 0x828E)) {
        if (p->count == 4) for (int i = 0; i < 4; ++i) info.cfa[i] = static_cast<int>(value(r, *p, i));
    }
    for (int i = 0; i < 4; ++i) if (info.cfa[i] < 0 || info.cfa[i] > 2) { error = "Unsupported CFA pattern"; return false; }
    // BlackLevel with BlackLevelRepeatDim (1x1 or 2x2).
    int bry = static_cast<int>(any(0xC619, 1, 0)), brx = static_cast<int>(any(0xC619, 1, 1));
    if (auto b = find(raw, 0xC61A)) {
        for (int y = 0; y < 2; ++y)
            for (int x = 0; x < 2; ++x) {
                uint32_t i = static_cast<uint32_t>((y % std::max(1, bry)) * std::max(1, brx) + x % std::max(1, brx));
                info.black[y * 2 + x] = static_cast<float>(i < b->count ? value(r, *b, i) : value(r, *b, 0));
            }
    }
    info.white = static_cast<float>(any(0xC61D, (1 << info.bps) - 1));
    info.activeTop = static_cast<int>(any(0xC68D, 0, 0));
    info.activeLeft = static_cast<int>(any(0xC68D, 0, 1));
    info.cropX = static_cast<int>(any(0xC61F, 0, 0));
    info.cropY = static_cast<int>(any(0xC61F, 0, 1));
    info.cropW = static_cast<int>(any(0xC620, info.width, 0));
    info.cropH = static_cast<int>(any(0xC620, info.height, 1));
    if (info.cropX < 0 || info.cropY < 0 || info.cropW <= 0 || info.cropH <= 0 ||
        info.activeLeft + info.cropX + info.cropW > info.width || info.activeTop + info.cropY + info.cropH > info.height) {
        info.cropX = info.cropY = info.activeLeft = info.activeTop = 0;
        info.cropW = info.width;
        info.cropH = info.height;
    }
    if (auto m = find(ifd0, 0xC621); m && m->count == 9) { for (int i = 0; i < 9; ++i) info.cm1[i] = value(r, *m, i); info.hasCM1 = true; }
    if (auto m = find(ifd0, 0xC622); m && m->count == 9) { for (int i = 0; i < 9; ++i) info.cm2[i] = value(r, *m, i); info.hasCM2 = true; }
    info.illum1 = static_cast<int>(num(ifd0, 0xC65A, 0));
    info.illum2 = static_cast<int>(num(ifd0, 0xC65B, 0));
    if (auto n = find(ifd0, 0xC628); n && n->count == 3) for (int i = 0; i < 3; ++i) info.neutral[i] = value(r, *n, i);
    if (auto a = find(ifd0, 0xC627); a && a->count == 3) for (int i = 0; i < 3; ++i) info.analog[i] = value(r, *a, i);
    info.baselineExposure = num(ifd0, 0xC62A, 0);
    info.focalMm = num(exif, 0x920A, 0);
    info.fNumber = num(exif, 0x829D, 0);
    info.focusM = num(exif, 0x9206, 0);
    if (auto l = find(exif, 0xA434); l && l->type == 2) {
        info.lensModel.assign(reinterpret_cast<const char*>(data + l->at), l->count);
        info.lensModel = info.lensModel.c_str();
        while (!info.lensModel.empty() && info.lensModel.back() == ' ') info.lensModel.pop_back();
    }
    // Opcode lists: big-endian opcodes; WarpRectilinear is id 1, GainMap is id 9. Video frames
    // carry both in OpcodeList3; stills keep the vignette's GainMap in OpcodeList2.
    for (uint16_t list : {uint16_t(0xC74E), uint16_t(0xC741)}) {
        auto o = find(raw, list) ? find(raw, list) : find(ifd0, list);
        if (!o || o->count < 4) continue;
        const Reader b{data + o->at, o->count, false};
        auto f32 = [&](size_t at) { float f; uint32_t v = b.u32(at); std::memcpy(&f, &v, 4); return f; };
        auto f64 = [&](size_t at) { double f; uint64_t v = static_cast<uint64_t>(b.u32(at)) << 32 | b.u32(at + 4); std::memcpy(&f, &v, 8); return f; };
        size_t at = 4;
        for (uint32_t k = 0, n = b.u32(0); k < n && b.ok(at, 16); ++k) {
            const uint32_t id = b.u32(at), bytes = b.u32(at + 12);
            const size_t body = at + 16;
            if (!b.ok(body, bytes)) break;
            at = body + bytes;
            if (id == 1 && bytes >= 4 && !info.warp.valid) {
                const uint32_t planes = b.u32(body);
                if ((planes == 1 || planes == 3) && bytes >= 4 + 48 * planes + 16) {
                    const size_t at0 = body + 4 + 48 * (planes == 3 ? 1 : 0);
                    LensWarp w;
                    for (int i = 0; i < 4; ++i) w.c[2 * i] = f64(at0 + 8 * i);   // terms in r^0, r^2, r^4, r^6
                    w.cx = f64(body + 4 + 48 * planes);
                    w.cy = f64(body + 4 + 48 * planes + 8);
                    bool sane = w.c[0] > 0.5 && w.c[0] < 2 && w.cx > 0.2 && w.cx < 0.8 && w.cy > 0.2 && w.cy < 0.8;
                    for (int i = 1; i < 4; ++i) sane = sane && std::fabs(w.c[2 * i]) < 2;
                    w.valid = sane;
                    if (sane) info.warp = w;
                }
                continue;
            }
            if (id != 9 || bytes < 76 || info.shading.valid()) continue;
            GainMap g;
            g.areaH = static_cast<int>(b.u32(body + 8)) - static_cast<int>(b.u32(body));
            g.areaW = static_cast<int>(b.u32(body + 12)) - static_cast<int>(b.u32(body + 4));
            g.rows = static_cast<int>(b.u32(body + 32));
            g.cols = static_cast<int>(b.u32(body + 36));
            g.planes = static_cast<int>(b.u32(body + 72));
            // Only what the fp writes: every pixel (pitch 1), the map spread over the whole area.
            const bool whole = b.u32(body + 24) == 1 && b.u32(body + 28) == 1;
            const size_t count = static_cast<size_t>(std::max(0, g.rows)) * std::max(0, g.cols) * std::max(0, g.planes);
            if (!whole || !g.valid() || g.rows > 512 || g.cols > 512 || bytes < 76 + 4 * count) continue;
            g.gain.resize(count);
            bool sane = true;
            for (size_t i = 0; i < count; ++i) {
                g.gain[i] = f32(body + 76 + 4 * i);
                if (!(g.gain[i] > 0.05f && g.gain[i] < 20.f)) sane = false;
            }
            if (sane) info.shading = std::move(g);
        }
    }
    info.iso = static_cast<int>(num(exif, 0x8827, num(ifd0, 0x8827, 0)));
    info.fps = num(ifd0, 0xC764, 0);
    if (auto m = find(ifd0, 0xC614)) info.model.assign(reinterpret_cast<const char*>(data + m->at), strnlen(reinterpret_cast<const char*>(data + m->at), m->count));
    if (!info.hasCM1) { error = "DNG has no ColorMatrix1"; return false; }
    if (info.compression != 1 && info.compression != 7) { error = "Unsupported DNG compression"; return false; }
    return true;
}

namespace {

void unpack_strip(const uint8_t* src, size_t bytes, const DngInfo& info, uint16_t* out, int y0, int y1) {
    const int W = info.width;
    if (info.bps == 12) {
        const size_t rowBytes = static_cast<size_t>(W) * 3 / 2;
        for (int y = y0; y < y1; ++y) {
            const uint8_t* s = src + rowBytes * y;
            uint16_t* o = out + static_cast<size_t>(W) * y;
            for (int x = 0; x + 1 < W; x += 2, s += 3) {
                o[x] = static_cast<uint16_t>(s[0] << 4 | s[1] >> 4);
                o[x + 1] = static_cast<uint16_t>((s[1] & 15) << 8 | s[2]);
            }
        }
    } else if (info.bps == 16) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x) {
                const uint8_t* s = src + 2 * (static_cast<size_t>(W) * y + x);
                out[static_cast<size_t>(W) * y + x] = static_cast<uint16_t>(s[0] << 8 | s[1]);
            }
    } else {
        // Generic MSB-first bit packing.
        const int b = info.bps;
        for (int y = y0; y < y1; ++y) {
            size_t bit = static_cast<size_t>(W) * y * b;
            for (int x = 0; x < W; ++x, bit += b) {
                uint32_t v = 0;
                for (int k = 0; k < b; ++k) {
                    size_t q = bit + k;
                    v = v << 1 | ((src[q >> 3] >> (7 - (q & 7))) & 1);
                }
                out[static_cast<size_t>(W) * y + x] = static_cast<uint16_t>(v);
            }
        }
    }
    (void)bytes;
}

template <class F>
void parallel(int n, int threads, F&& f) {
    if (threads <= 1 || n <= 1) { for (int i = 0; i < n; ++i) f(i); return; }
    threads = std::min(threads, n);
    std::vector<std::thread> pool;
    std::atomic<int> next{0};
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&] { for (int i; (i = next.fetch_add(1)) < n;) f(i); });
    for (auto& t : pool) t.join();
}

}  // namespace

bool decode_raw(const uint8_t* data, size_t size, const DngInfo& info, uint16_t* out, int threads,
                std::string& error) {
    (void)size;
    const int W = info.width, H = info.height;
    if (info.compression == 1) {
        if (info.offsets.size() != 1) {
            // Multi-strip: rows are contiguous in file order when strips are contiguous.
            for (size_t i = 1; i < info.offsets.size(); ++i)
                if (info.offsets[i] != info.offsets[i - 1] + info.counts[i - 1]) { error = "Non-contiguous strips unsupported"; return false; }
        }
        size_t need = (static_cast<size_t>(W) * H * info.bps + 7) / 8;
        uint64_t total = 0;
        for (auto c : info.counts) total += c;
        if (total < need) { error = "Raw strip too short"; return false; }
        const uint8_t* src = data + info.offsets[0];
        const int bands = threads > 1 ? threads * 2 : 1;
        parallel(bands, threads, [&](int b) {
            unpack_strip(src, total, info, out, H * b / bands, H * (b + 1) / bands);
        });
        return true;
    }
    if (info.tileW <= 0 || info.tileH <= 0) { error = "Compressed DNG without tiles"; return false; }
    const int tx = (W + info.tileW - 1) / info.tileW, ty = (H + info.tileH - 1) / info.tileH;
    if (static_cast<int>(info.offsets.size()) != tx * ty) { error = "Tile count mismatch"; return false; }
    std::atomic<bool> failed{false};
    std::string firstError;
    std::mutex m;
    parallel(tx * ty, threads, [&](int t) {
        if (failed) return;
        int x0 = (t % tx) * info.tileW, y0 = (t / tx) * info.tileH;
        std::string e;
        if (!lj92_decode(data + info.offsets[t], static_cast<size_t>(info.counts[t]),
                         out + static_cast<size_t>(y0) * W + x0, W, std::min(info.tileW, W - x0),
                         std::min(info.tileH, H - y0), info.tileW, info.tileH, e)) {
            std::lock_guard<std::mutex> g(m);
            if (!failed) { failed = true; firstError = "Tile " + std::to_string(t) + ": " + e; }
        }
    });
    if (failed) { error = firstError; return false; }
    return true;
}

}  // namespace sfp
