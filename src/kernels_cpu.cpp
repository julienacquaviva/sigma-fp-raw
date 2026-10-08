// CPU development: the kernels of kernels.cu compiled as plain C++ and run over the image
// rows on all processor cores. Used when there is no NVIDIA CUDA driver (macOS, AMD and
// Intel graphics) or when SFP_CPU=1 asks for it; same picture as the GPU path.
#include <math.h>
#include <stddef.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "develop_cpu.h"

namespace sfp {
namespace cpu {

#include "kernels.cu"

namespace {

// Runs job(band) for band = 0..bands-1 on the calling thread and the workers.
class Pool {
public:
    Pool() {
        const unsigned n = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned i = 1; i < std::min(n, 64u); ++i) workers.emplace_back([this] { loop(); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> l(m);
            stop = true;
        }
        wake.notify_all();
        for (auto& t : workers) t.join();
    }
    void run(int bands, const std::function<void(int)>& fn) {
        {
            std::lock_guard<std::mutex> l(m);
            job = &fn;
            count = bands;
            next = 0;
            busy = static_cast<int>(workers.size());
            ++generation;
        }
        wake.notify_all();
        for (int b; (b = next.fetch_add(1)) < bands;) fn(b);
        std::unique_lock<std::mutex> l(m);
        done.wait(l, [&] { return busy == 0; });
        job = nullptr;
    }

private:
    void loop() {
        unsigned seen = 0;
        for (;;) {
            const std::function<void(int)>* fn;
            int bands;
            {
                std::unique_lock<std::mutex> l(m);
                wake.wait(l, [&] { return stop || generation != seen; });
                if (stop) return;
                seen = generation;
                fn = job;
                bands = count;
            }
            for (int b; (b = next.fetch_add(1)) < bands;) (*fn)(b);
            std::lock_guard<std::mutex> l(m);
            if (--busy == 0) done.notify_all();
        }
    }
    std::vector<std::thread> workers;
    std::mutex m;
    std::condition_variable wake, done;
    const std::function<void(int)>* job = nullptr;
    std::atomic<int> next{0};
    int count = 0, busy = 0;
    unsigned generation = 0;
    bool stop = false;
};

// f(x, y) for every pixel of a w x h image, rows shared out in bands of 16.
template <class F>
void each_pixel(Pool& pool, int w, int h, F&& f) {
    const int rows = 16, bands = (h + rows - 1) / rows;
    pool.run(bands, [&](int b) {
        const int y1 = std::min(h, (b + 1) * rows);
        for (int y = b * rows; y < y1; ++y)
            for (int x = 0; x < w; ++x) f(x, y);
    });
}

struct State {
    std::mutex m;
    Pool pool;
    std::vector<unsigned short> raw2;
    std::vector<float> cfa, vh, lpf, pq, R, G, B, T[6];
    const float* planes[3] = {nullptr, nullptr, nullptr};
    size_t cap = 0;
    // Demosaic reuse key.
    bool valid = false;
    uint64_t frameId = 0;
    int quality = -1;
    float lastDezig = -1;
    PrepParams last{};
};

}  // namespace
}  // namespace cpu

bool develop_cpu(const Frame& f, int quality, const PrepParams& pp, float dz, const DevelopParams& dp, void* out,
                 std::string& e, DevelopTiming* timing, const void* source, int sourceRowBytes) {
    using namespace cpu;
    static State* st = new State;   // kept for the life of the process (no thread joins at unload)
    State& x = *st;
    std::lock_guard<std::mutex> lock(x.m);
    const int W = source ? dp.W : pp.W, H = source ? dp.H : pp.H;
    const size_t n = static_cast<size_t>(W) * H;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        if (x.cap < n) {
            x.valid = false;
            x.raw2.resize(n);
            for (auto* v : {&x.cfa, &x.vh, &x.lpf, &x.pq, &x.R, &x.G, &x.B, &x.T[0], &x.T[1], &x.T[2], &x.T[3], &x.T[4], &x.T[5]}) v->resize(n);
            x.cap = n;
        }
    } catch (const std::bad_alloc&) {
        x.cap = 0;
        e = "Not enough memory";
        return false;
    }
    const bool reuse = !source && x.valid && x.frameId == f.id && x.quality == quality && x.lastDezig == dz && !std::memcmp(&x.last, &pp, sizeof pp);
    if (source) {
        x.valid = false;
        const float* in = static_cast<const float*>(source);
        float *R = x.R.data(), *G = x.G.data(), *B = x.B.data();
        each_pixel(x.pool, W, H, [&](int px, int py) { k_unpack(px, py, in, R, G, B, W, H, sourceRowBytes); });
        x.planes[0] = R; x.planes[1] = G; x.planes[2] = B;
    } else if (!reuse) {
        x.valid = false;
        const unsigned short* src = f.raw.data();
        float *cfa = x.cfa.data(), *vh = x.vh.data(), *lpf = x.lpf.data(), *pq = x.pq.data();
        float *R = x.R.data(), *G = x.G.data(), *B = x.B.data();
        if (pp.hotPixels) {
            unsigned short* o = x.raw2.data();
            each_pixel(x.pool, W, H, [&](int px, int py) { k_hot(px, py, src, o, pp); });
            src = o;
        }
        each_pixel(x.pool, W, H, [&](int px, int py) { k_prep(px, py, src, cfa, pp); });
        if (quality == 1) {
            each_pixel(x.pool, W / 2, H / 2, [&](int px, int py) { k_half(px, py, cfa, R, G, B, pp); });
        } else {
            each_pixel(x.pool, W, H, [&](int px, int py) { k_vh(px, py, cfa, vh, W, H); });
            each_pixel(x.pool, W, H, [&](int px, int py) { k_lpf(px, py, cfa, lpf, pp); });
            each_pixel(x.pool, W, H, [&](int px, int py) { k_green(px, py, cfa, vh, lpf, G, pp); });
            each_pixel(x.pool, W, H, [&](int px, int py) { k_pq(px, py, cfa, pq, pp); });
            each_pixel(x.pool, W, H, [&](int px, int py) { k_rb_diag(px, py, cfa, G, pq, R, B, pp); });
            each_pixel(x.pool, W, H, [&](int px, int py) { k_rb_g(px, py, G, vh, R, B, pp); });
        }
        x.planes[0] = R; x.planes[1] = G; x.planes[2] = B;
        if (dz > 0.f) {
            // Demosaic scratch (cfa, vh, lpf) is free now: de-zigzag into it.
            const DezigParams zp{W, H, dz, 12.0f, 0.08f, 0.45f, 2e-5f};
            float* T[6];
            for (int i = 0; i < 6; ++i) T[i] = x.T[i].data();
            each_pixel(x.pool, W, H, [&](int px, int py) { k_tensor(px, py, R, G, B, T[0], T[1], T[2], W, H); });
            each_pixel(x.pool, W, H, [&](int px, int py) { k_blur3(px, py, T[0], T[1], T[2], T[3], T[4], T[5], W, H, 0); });
            each_pixel(x.pool, W, H, [&](int px, int py) { k_blur3(px, py, T[3], T[4], T[5], T[0], T[1], T[2], W, H, 1); });
            each_pixel(x.pool, W, H, [&](int px, int py) { k_dezig(px, py, R, G, B, T[0], T[1], T[2], cfa, vh, lpf, zp); });
            x.planes[0] = cfa; x.planes[1] = vh; x.planes[2] = lpf;
        }
        x.lastDezig = dz;
        x.valid = true;
        x.frameId = f.id;
        x.quality = quality;
        x.last = pp;
    }
    if (timing) timing->demosaicReused = reuse;
    const float *P0 = x.planes[0], *P1 = x.planes[1], *P2 = x.planes[2];
    float* o = static_cast<float*>(out);
    each_pixel(x.pool, dp.outW, dp.outH, [&](int px, int py) { k_develop(px, py, P0, P1, P2, o, dp); });
    if (timing) timing->gpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace sfp
