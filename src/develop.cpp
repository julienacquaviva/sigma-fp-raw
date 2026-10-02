#include "develop.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "develop_cpu.h"
#include "platform.h"

#ifdef __APPLE__
static const unsigned char kKernelsPtx[1] = {0};   // no CUDA on macOS
#else
#include "kernels_ptx.h"
#endif

namespace sfp {

// 3K frames (sensor mode M98, /2 binned readout) are the only ones with the row-pair
// offset and its stair-step aliasing; both corrections apply to them only.
static bool is_3k(const DngInfo& info) { return info.width == 3024 && info.height == 2010; }

double effective_row_phase(const RawSettings& s, const DngInfo& info) {
    // G2/B rows sit 0.5 px off the Bayer grid (A001_065 raw: G2-G1 +0.51 px);
    // recommended -0.125 plane pitch re-spaces the row pairs.
    return is_3k(info) ? std::clamp(s.rowPhase, -0.5, 0.5) : 0.0;
}

double effective_dezigzag(const RawSettings& s, const DngInfo& info) {
    return is_3k(info) ? std::clamp(s.deZigzag, 0.0, 100.0) : 0.0;
}

bool preset_temp_tint(WhiteBalance wb, double& temp, double& tint) {
    switch (wb) {
        case WhiteBalance::Daylight: temp = 5500; tint = 10; return true;
        case WhiteBalance::Cloudy: temp = 6500; tint = 10; return true;
        case WhiteBalance::Shade: temp = 7500; tint = 10; return true;
        case WhiteBalance::Tungsten: temp = 2850; tint = 0; return true;
        case WhiteBalance::Fluorescent: temp = 3800; tint = 21; return true;
        case WhiteBalance::Flash: temp = 5500; tint = 0; return true;
        default: return false;
    }
}

namespace {

struct Ctx {
    CUcontext ctx = nullptr;
    CUmodule mod = nullptr;
    CUfunction hot, prep, vh, lpf, green, pq, rbDiag, rbG, half, tensor, blur3, dezig, dev;
    CUdeviceptr T[6] = {0, 0, 0, 0, 0, 0};   // structure tensor planes (+ blur temp)
    CUdeviceptr planes[3] = {0, 0, 0};   // develop input (demosaic or de-zigzag output)
    float lastDezig = -1;
    CUdeviceptr raw = 0, raw2 = 0, cfa = 0, vhb = 0, lpfb = 0, pqb = 0, R = 0, G = 0, B = 0;
    size_t cap = 0;
    CUdeviceptr out = 0;
    size_t outCap = 0;
    // Demosaic reuse key.
    bool valid = false;
    uint64_t frameId = 0;
    int quality = -1;
    PrepParams last{};
    CUstream lastStream = nullptr;
    std::vector<char> staging;
    void* pinned = nullptr;       // page-locked upload staging (fast DMA)
    size_t pinnedCap = 0;
    std::mutex m;
};

bool check(CUresult r, const char* what, std::string& e) {
    if (r == 0) return true;
    e = std::string(what) + ": " + cuda_error(r);
    return false;
}

}  // namespace

struct Developer::Impl {
    std::mutex m;
    std::map<CUcontext, std::unique_ptr<Ctx>> ctxs;
    CUcontext primary = nullptr;

    std::map<CUcontext, std::string> unusable;   // contexts whose GPU cannot run the kernels (too old)

    Ctx* context(CUcontext c, std::string& e) {
        std::lock_guard<std::mutex> l(m);
        auto bad = unusable.find(c);
        if (bad != unusable.end()) { e = bad->second; return nullptr; }
        auto& slot = ctxs[c];
        if (slot) return slot.get();
        auto x = std::make_unique<Ctx>();
        auto& cu = cuda();
        x->ctx = c;
        // Test hook: SFP_TEST_NO_KERNELS=1 behaves like a GPU that cannot load the kernels.
        if (os::env("SFP_TEST_NO_KERNELS") == "1" || !check(cu.cuModuleLoadData(&x->mod, kKernelsPtx), "Load GPU kernels", e)) {
            ctxs.erase(c);
            unusable[c] = e;
            return nullptr;
        }
        struct { CUfunction* f; const char* n; } fns[] = {
            {&x->hot, "k_hot"}, {&x->prep, "k_prep"}, {&x->vh, "k_vh"}, {&x->lpf, "k_lpf"}, {&x->green, "k_green"}, {&x->pq, "k_pq"},
            {&x->rbDiag, "k_rb_diag"}, {&x->rbG, "k_rb_g"}, {&x->half, "k_half"}, {&x->tensor, "k_tensor"}, {&x->blur3, "k_blur3"}, {&x->dezig, "k_dezig"}, {&x->dev, "k_develop"}};
        for (auto& f : fns)
            if (!check(cu.cuModuleGetFunction(f.f, x->mod, f.n), f.n, e)) { ctxs.erase(c); return nullptr; }
        slot = std::move(x);
        return slot.get();
    }
};

Developer::Developer() : d(std::make_shared<Impl>()) {}
Developer& Developer::get() {
    static Developer dev;
    return dev;
}
const char* Developer::backend() { return os::env("SFP_CPU") != "1" && cuda().ok ? "CUDA (GPU)" : "CPU"; }

bool Developer::develop(const Frame& f, const RawSettings& s, CUstream stream, const Target& t,
                        std::string& e, DevelopTiming* timing) {
    const DngInfo& info = f.info;
    const int W = info.width, H = info.height;
    if (W < 16 || H < 16 || f.raw.size() != static_cast<size_t>(W) * H) { e = "Bad frame"; return false; }
    if (t.width <= 0 || t.height <= 0 || t.rodW <= 0 || t.rodH <= 0 || (!t.device && !t.host)) { e = "Bad output image"; return false; }
    // Colour.
    double temp = s.colorTemp, tint = s.tint;
    bool asShot = s.whiteBalance == WhiteBalance::AsShot;
    if (!asShot && s.whiteBalance != WhiteBalance::Custom) preset_temp_tint(s.whiteBalance, temp, tint);
    ColorSetup cs = color_setup(info, asShot, temp, tint, s.colorSpace);
    const float norm = std::max({cs.wb[0], cs.wb[1], cs.wb[2]});
    PrepParams pp{};
    pp.W = W; pp.H = H;
    for (int k = 0; k < 4; ++k) {
        pp.cfa[k] = info.cfa[k];
        pp.black[k] = info.black[k];
        float range = std::max(1.f, info.white - info.black[k]);
        pp.scale[k] = cs.wb[info.cfa[k]] / range / norm;
    }
    pp.phase = static_cast<float>(effective_row_phase(s, info));
    pp.white = info.white;
    pp.hotPixels = s.hotPixels ? 1 : 0;
    // Raw below black reaches -black/(white-black) (about -0.067 on the fp): keep all of it positive.
    pp.pedestal = 0.1f;
    const float dz = static_cast<float>(effective_dezigzag(s, info) / 100.0);
    // Develop parameters.
    DevelopParams dp{};
    dp.W = W; dp.H = H;
    dp.cropX = info.activeLeft + info.cropX;
    dp.cropY = info.activeTop + info.cropY;
    dp.cropW = info.cropW;
    dp.cropH = info.cropH;
    dp.outW = t.width; dp.outH = t.height;
    // Host images may have padded or negative strides: develop packed, copy rows after.
    dp.rowBytes = t.device ? t.rowBytes : t.width * 16;   // (the CPU path writes the host rows directly)
    dp.boundsX = t.boundsX; dp.boundsY = t.boundsY; dp.rodH = t.rodH;
    double sx = static_cast<double>(t.rodW) / dp.cropW, sy = static_cast<double>(t.rodH) / dp.cropH;
    switch (s.fit) {
        case Fit::Fill: sx = sy = std::max(sx, sy); break;
        case Fit::Stretch: break;
        case Fit::Native: sx = s.renderScaleX; sy = s.renderScaleY; break;
        default: sx = sy = std::min(sx, sy); break;
    }
    dp.scaleX = static_cast<float>(sx);
    dp.scaleY = static_cast<float>(sy);
    dp.offX = static_cast<float>((t.rodW - dp.cropW * sx) / 2);
    dp.offY = static_cast<float>((t.rodH - dp.cropH * sy) / 2);
    dp.nearest = std::fabs(sx - 1) < 1e-6 && std::fabs(sy - 1) < 1e-6 &&
                 std::fabs(dp.offX - std::round(dp.offX)) < 1e-4 && std::fabs(dp.offY - std::round(dp.offY)) < 1e-4;
    dp.stab = s.stab;
    if (dp.stab.on) dp.nearest = 0;
    dp.sharpen = static_cast<float>(std::clamp(s.sharpness, 0.0, 100.0) / 100.0 * 2.0);
    dp.norm = norm;
    dp.clip = std::min({cs.wb[0], cs.wb[1], cs.wb[2]});
    for (int i = 0; i < 3; ++i) dp.chanClip[i] = cs.wb[i];
    dp.highlightRecovery = s.highlightRecovery;
    for (int i = 0; i < 9; ++i) dp.m[i] = cs.matrix[i];
    dp.gainLin = static_cast<float>(std::exp2(info.baselineExposure + std::clamp(s.exposure, -10.0, 10.0)));
    for (int i = 0; i < 3; ++i) dp.lum[i] = cs.lum[i];
    dp.contrast = static_cast<float>(std::exp2((std::clamp(s.contrast, 0.0, 100.0) - 50.0) / 50.0));
    dp.midtones = static_cast<float>(std::clamp(s.midtones, -100.0, 100.0) / 100.0);
    dp.highlights = static_cast<float>(std::clamp(s.highlights, -100.0, 100.0) / 100.0);
    dp.shadows = static_cast<float>(std::clamp(s.shadows, -100.0, 100.0) / 100.0);
    dp.toneActive = dp.contrast != 1.f || dp.midtones != 0.f || dp.highlights != 0.f || dp.shadows != 0.f;
    dp.colorBoost = static_cast<float>(std::clamp(s.colorBoost, 0.0, 100.0) / 100.0);
    dp.saturation = static_cast<float>(std::clamp(s.saturation, 0.0, 100.0) / 50.0);
    dp.gamutMap = s.gamutMapping;
    dp.preToneCurve = s.preToneCurve;
    dp.softClip = s.softClip;
    dp.gamma = static_cast<int>(s.gamma);
    dp.lift = static_cast<float>(std::clamp(s.lift, -100.0, 100.0) / 100.0 * 0.25);
    dp.gain = static_cast<float>(1.0 + std::clamp(s.gain, -100.0, 100.0) / 100.0);
    dp.pedestal = pp.pedestal;
    // Without an NVIDIA CUDA driver (or with SFP_CPU=1) host images are developed on the CPU.
    static const bool forceCpu = os::env("SFP_CPU") == "1";
    auto& cu = cuda();
    if (!t.device && (forceCpu || !cu.ok)) {
        dp.rowBytes = t.rowBytes;
        return develop_cpu(f, s.decodeQuality, pp, dz, dp, t.host, e, timing);
    }
    if (!cu.ok) { e = cu.error; return false; }
    // Context: the host's current context for device images, else the primary context.
    CUcontext c = nullptr;
    bool pushed = false;
    if (t.device) {
        if (!check(cu.cuCtxGetCurrent(&c), "CUDA context", e)) return false;
        if (!c) { e = "The host supplied CUDA images without a current CUDA context"; return false; }
    } else {
        std::lock_guard<std::mutex> l(d->m);
        if (!d->primary) {
            CUdevice dev;
            if (!check(cu.cuDeviceGet(&dev, 0), "CUDA device", e) ||
                !check(cu.cuDevicePrimaryCtxRetain(&d->primary, dev), "CUDA context", e)) return false;
        }
        c = d->primary;
    }
    if (!t.device) {
        if (!check(cu.cuCtxPushCurrent(c), "CUDA context", e)) return false;
        pushed = true;
    }
    struct Pop { bool on; ~Pop() { if (on) { CUcontext x; cuda().cuCtxPopCurrent(&x); } } } pop{pushed};
    Ctx* x = d->context(c, e);
    if (!x) {
        // The GPU cannot run the kernels (older than the PTX target): develop on the CPU.
        if (!t.device) {
            dp.rowBytes = t.rowBytes;
            return develop_cpu(f, s.decodeQuality, pp, dz, dp, t.host, e, timing);
        }
        const size_t packed = static_cast<size_t>(t.width) * 16;
        std::vector<char> img(packed * t.height);
        dp.rowBytes = static_cast<int>(packed);
        if (!develop_cpu(f, s.decodeQuality, pp, dz, dp, img.data(), e, timing)) return false;
        for (int y = 0; y < t.height; ++y)
            if (!check(cu.cuMemcpyHtoDAsync(t.device + static_cast<long long>(y) * t.rowBytes, img.data() + y * packed, packed, stream), "Upload image", e)) return false;
        return check(cu.cuStreamSynchronize(stream), "Upload image", e);
    }
    std::lock_guard<std::mutex> lock(x->m);
    // Buffers are shared by all renders in this context; serialise across streams.
    if (x->lastStream != stream && x->lastStream) cu.cuStreamSynchronize(x->lastStream);
    x->lastStream = stream;
    const size_t n = static_cast<size_t>(W) * H;
    if (x->cap < n) {
        for (CUdeviceptr* p : {&x->raw, &x->raw2, &x->cfa, &x->vhb, &x->lpfb, &x->pqb, &x->R, &x->G, &x->B, &x->T[0], &x->T[1], &x->T[2], &x->T[3], &x->T[4], &x->T[5]})
            if (*p) { cu.cuMemFree(*p); *p = 0; }
        x->cap = 0;
        x->valid = false;
        if (!check(cu.cuMemAlloc(&x->raw, n * 2), "GPU memory", e) || !check(cu.cuMemAlloc(&x->raw2, n * 2), "GPU memory", e)) return false;
        for (CUdeviceptr* p : {&x->cfa, &x->vhb, &x->lpfb, &x->pqb, &x->R, &x->G, &x->B, &x->T[0], &x->T[1], &x->T[2], &x->T[3], &x->T[4], &x->T[5]})
            if (!check(cu.cuMemAlloc(p, n * 4), "GPU memory", e)) return false;
        x->cap = n;
    }
    auto t0 = std::chrono::steady_clock::now();
    const unsigned bx = 32, by = 8;
    auto launch = [&](CUfunction fn, unsigned w, unsigned h, void** args, const char* name) {
        return check(cu.cuLaunchKernel(fn, (w + bx - 1) / bx, (h + by - 1) / by, 1, bx, by, 1, 0, stream, args, nullptr), name, e);
    };
    bool reuse = x->valid && x->frameId == f.id && x->quality == s.decodeQuality && x->lastDezig == dz && !std::memcmp(&x->last, &pp, sizeof pp);
    if (!reuse) {
        x->valid = false;
        // Stage through page-locked memory: the previous upload from it has completed
        // once the stream has drained, so wait for the stream first.
        if (x->pinnedCap < n * 2) {
            if (x->pinned) cu.cuMemFreeHost(x->pinned);
            x->pinned = nullptr;
            x->pinnedCap = 0;
            if (cu.cuMemAllocHost(&x->pinned, n * 2) == 0) x->pinnedCap = n * 2;
        }
        if (x->pinned) {
            cu.cuStreamSynchronize(stream);
            std::memcpy(x->pinned, f.raw.data(), n * 2);
            if (!check(cu.cuMemcpyHtoDAsync(x->raw, x->pinned, n * 2, stream), "Upload raw frame", e)) return false;
        } else if (!check(cu.cuMemcpyHtoDAsync(x->raw, f.raw.data(), n * 2, stream), "Upload raw frame", e)) return false;
        if (timing) { cu.cuStreamSynchronize(stream); timing->uploadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
        CUdeviceptr src = x->raw;
        if (pp.hotPixels) {
            void* aHot[] = {&x->raw, &x->raw2, &pp};
            if (!launch(x->hot, W, H, aHot, "k_hot")) return false;
            src = x->raw2;
        }
        void* aPrep[] = {&src, &x->cfa, &pp};
        if (!launch(x->prep, W, H, aPrep, "k_prep")) return false;
        if (s.decodeQuality == 1) {
            void* a[] = {&x->cfa, &x->R, &x->G, &x->B, &pp};
            if (!launch(x->half, W / 2, H / 2, a, "k_half")) return false;
        } else {
            void* aVh[] = {&x->cfa, &x->vhb, &pp.W, &pp.H};
            void* aLpf[] = {&x->cfa, &x->lpfb, &pp};
            void* aGreen[] = {&x->cfa, &x->vhb, &x->lpfb, &x->G, &pp};
            void* aPq[] = {&x->cfa, &x->pqb, &pp};
            void* aDiag[] = {&x->cfa, &x->G, &x->pqb, &x->R, &x->B, &pp};
            void* aRbG[] = {&x->G, &x->vhb, &x->R, &x->B, &pp};
            if (!launch(x->vh, W, H, aVh, "k_vh") || !launch(x->lpf, W, H, aLpf, "k_lpf") ||
                !launch(x->green, W, H, aGreen, "k_green") || !launch(x->pq, W, H, aPq, "k_pq") ||
                !launch(x->rbDiag, W, H, aDiag, "k_rb_diag") || !launch(x->rbG, W, H, aRbG, "k_rb_g")) return false;
        }
        x->planes[0] = x->R; x->planes[1] = x->G; x->planes[2] = x->B;
        if (dz > 0.f) {
            // Demosaic scratch (cfa, vh, lpf) is free now: de-zigzag into it.
            // Strength scales the blend; the averaging length adapts to the edge slope (<= 12 px).
            DezigParams zp{W, H, dz, 12.0f, 0.08f, 0.45f, 2e-5f};
            int d0 = 0, d1 = 1;
            void* aT[] = {&x->R, &x->G, &x->B, &x->T[0], &x->T[1], &x->T[2], &pp.W, &pp.H};
            void* aBh[] = {&x->T[0], &x->T[1], &x->T[2], &x->T[3], &x->T[4], &x->T[5], &pp.W, &pp.H, &d0};
            void* aBv[] = {&x->T[3], &x->T[4], &x->T[5], &x->T[0], &x->T[1], &x->T[2], &pp.W, &pp.H, &d1};
            void* a[] = {&x->R, &x->G, &x->B, &x->T[0], &x->T[1], &x->T[2], &x->cfa, &x->vhb, &x->lpfb, &zp};
            if (!launch(x->tensor, W, H, aT, "k_tensor") || !launch(x->blur3, W, H, aBh, "k_blur3") ||
                !launch(x->blur3, W, H, aBv, "k_blur3") || !launch(x->dezig, W, H, a, "k_dezig")) return false;
            x->planes[0] = x->cfa; x->planes[1] = x->vhb; x->planes[2] = x->lpfb;
        }
        x->lastDezig = dz;
        x->valid = true;
        x->frameId = f.id;
        x->quality = s.decodeQuality;
        x->last = pp;
    }
    if (timing) timing->demosaicReused = reuse;
    CUdeviceptr out = t.device;
    const size_t outBytes = static_cast<size_t>(t.width) * 16 * t.height;
    if (!out) {
        if (x->outCap < outBytes) {
            if (x->out) cu.cuMemFree(x->out);
            x->out = 0;
            x->outCap = 0;
            if (!check(cu.cuMemAlloc(&x->out, outBytes), "GPU memory", e)) return false;
            x->outCap = outBytes;
        }
        out = x->out;
    }
    void* aDev[] = {&x->planes[0], &x->planes[1], &x->planes[2], &out, &dp};
    if (!check(cu.cuLaunchKernel(x->dev, (t.width + 15) / 16, (t.height + 15) / 16, 1, 16, 16, 1, 0, stream, aDev, nullptr), "k_develop", e)) return false;
    if (!t.device) {
        const size_t packed = static_cast<size_t>(t.width) * 16;
        if (t.rowBytes == static_cast<int>(packed)) {
            if (!check(cu.cuMemcpyDtoHAsync(t.host, out, outBytes, stream), "Download image", e)) return false;
            if (!check(cu.cuStreamSynchronize(stream), "GPU processing", e)) return false;
        } else {
            x->staging.resize(outBytes);
            if (!check(cu.cuMemcpyDtoHAsync(x->staging.data(), out, outBytes, stream), "Download image", e)) return false;
            if (!check(cu.cuStreamSynchronize(stream), "GPU processing", e)) return false;
            for (int y = 0; y < t.height; ++y)
                std::memcpy(static_cast<char*>(t.host) + static_cast<long long>(y) * t.rowBytes, x->staging.data() + y * packed, packed);
        }
    } else if (!stream || timing) {
        if (!check(cu.cuStreamSynchronize(stream), "GPU processing", e)) return false;
    }
    if (timing) timing->gpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace sfp
