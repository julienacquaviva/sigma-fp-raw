#include "develop.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "develop_cpu.h"
#include "develop_metal.h"
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
static bool binned_3k(const RawSettings& s, const DngInfo& info) { return is_3k(info) && s.binned != 0; }

double effective_row_phase(const RawSettings& s, const DngInfo& info) {
    // G2/B rows sit 0.5 px off the Bayer grid (A001_065 raw: G2-G1 +0.51 px);
    // recommended -0.125 plane pitch re-spaces the row pairs.
    return binned_3k(s, info) ? std::clamp(s.rowPhase, -0.5, 0.5) : 0.0;
}

double effective_dezigzag(const RawSettings& s, const DngInfo& info) {
    return binned_3k(s, info) ? std::clamp(s.deZigzag, 0.0, 100.0) : 0.0;
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
    CUfunction hot, prep, vh, lpf, green, pq, rbDiag, rbG, half, tensor, blur3, dezig, dev, unpack;
    CUdeviceptr src = 0;         // the host's picture, uploaded (pass-through with host images)
    size_t srcCap = 0;
    CUdeviceptr T[6] = {0, 0, 0, 0, 0, 0};   // structure tensor planes (+ blur temp)
    CUdeviceptr planes[3] = {0, 0, 0};   // develop input (demosaic or de-zigzag output)
    float lastDezig = -1;
    CUdeviceptr raw = 0, raw2 = 0, cfa = 0, vhb = 0, lpfb = 0, pqb = 0, R = 0, G = 0, B = 0;
    size_t cap = 0;
    size_t planeCap = 0;         // R, G, B alone (the host's own picture needs no more)
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
            {&x->rbDiag, "k_rb_diag"}, {&x->rbG, "k_rb_g"}, {&x->half, "k_half"}, {&x->tensor, "k_tensor"}, {&x->blur3, "k_blur3"}, {&x->dezig, "k_dezig"}, {&x->dev, "k_develop"}, {&x->unpack, "k_unpack"}};
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
static std::atomic<bool> gForceCpu{false};
void Developer::force_cpu(bool on) { gForceCpu = on; }
const char* Developer::backend() {
    if (os::env("SFP_CPU") == "1" || gForceCpu) return "CPU";
    return cuda().ok ? "CUDA (GPU)" : metal_available() ? "Metal (GPU)" : "CPU";
}

// The brightness plane of a shading map (over the whole 6048 x 4032 sensor) as a radial gain
// 1 + a0 r^2 + a1 r^4 + a2 r^6 + a3 r^8, r = 1 at the sensor's corner (least squares).
static void fit_vignette(const GainMap& g, float a[4]) {
    double M[4][5] = {};
    const double R2 = 0.25 * (6048.0 * 6048.0 + 4032.0 * 4032.0);
    for (int i = 0; i < g.rows; ++i)
        for (int j = 0; j < g.cols; ++j) {
            const double x = (static_cast<double>(j) / (g.cols - 1) - 0.5) * 6048, y = (static_cast<double>(i) / (g.rows - 1) - 0.5) * 4032;
            const double r2 = (x * x + y * y) / R2;
            const double b = g.gain[(static_cast<size_t>(i) * g.cols + j) * g.planes + (g.planes == 3 ? 1 : 0)] - 1.0;
            double basis[4] = {r2, r2 * r2, r2 * r2 * r2, r2 * r2 * r2 * r2};
            for (int u = 0; u < 4; ++u) {
                for (int v = 0; v < 4; ++v) M[u][v] += basis[u] * basis[v];
                M[u][4] += basis[u] * b;
            }
        }
    for (int c = 0; c < 4; ++c) {   // Gauss-Jordan with pivoting
        int piv = c;
        for (int r = c + 1; r < 4; ++r) if (std::fabs(M[r][c]) > std::fabs(M[piv][c])) piv = r;
        for (int k = 0; k < 5; ++k) std::swap(M[c][k], M[piv][k]);
        if (std::fabs(M[c][c]) < 1e-18) { for (int k = 0; k < 4; ++k) a[k] = 0; return; }
        for (int r = 0; r < 4; ++r) {
            if (r == c) continue;
            const double f = M[r][c] / M[c][c];
            for (int k = c; k < 5; ++k) M[r][k] -= f * M[c][k];
        }
    }
    for (int k = 0; k < 4; ++k) a[k] = static_cast<float>(M[k][4] / M[k][k]);
}

// Where an output position is read in the raster (the kernel's chain without the black borders).
static void out_to_source(const DevelopParams& p, float ix, float iy, float& sx, float& sy) {
    if (p.xform) xform_map(p, ix, iy, &ix, &iy);
    sx = (ix - p.offX) / p.scaleX + p.cropX;
    sy = (iy - p.offY) / p.scaleY + p.cropY;
    if (p.stab.on) stab_map(p.stab, sx, sy, &sx, &sy);
    if (p.lens.on) lens_map(p.lens, sx, sy, &sx, &sy);
}

// The kernel and how far it must be widened: source pixels per output pixel at the frame
// centre (the fit, the Transform zoom, the stabiliser's zoom; perspective and rolling shutter
// change it only a little across the frame).
void resample_scale(const RawSettings& s, DevelopParams& p) {
    p.resampler = std::clamp(s.resampler, 0, 3);
    p.kx = p.ky = 1.f;
    if (p.resampler == kResampleBilinear) return;
    const float cx = 0.5f * p.rodW, cy = 0.5f * p.rodH;
    float x0, y0, x1, y1, x2, y2;
    out_to_source(p, cx, cy, x0, y0);
    out_to_source(p, cx + 1.f, cy, x1, y1);
    out_to_source(p, cx, cy + 1.f, x2, y2);
    const float kx = std::sqrt((x1 - x0) * (x1 - x0) + (x2 - x0) * (x2 - x0));
    const float ky = std::sqrt((y1 - y0) * (y1 - y0) + (y2 - y0) * (y2 - y0));
    // Within 0.1 % of 1:1 counts as 1:1 (the finite differences are not exact).
    p.kx = std::isfinite(kx) && kx > 1.001f ? std::min(kx, 16.f) : 1.f;
    p.ky = std::isfinite(ky) && ky > 1.001f ? std::min(ky, 16.f) : 1.f;
}

void frame_geometry(int cropW, int cropH, const RawSettings& s, const Target& t, DevelopParams& dp) {
    dp.cropW = cropW; dp.cropH = cropH;
    dp.rodW = t.rodW; dp.rodH = t.rodH;
    double sx = static_cast<double>(t.rodW) / cropW, sy = static_cast<double>(t.rodH) / cropH;
    switch (s.fit) {
        case Fit::Fill: sx = sy = std::max(sx, sy); break;
        case Fit::Stretch: break;
        case Fit::Native: sx = s.renderScaleX; sy = s.renderScaleY; break;
        case Fit::FitWidth: sx = sy = (t.compW > 0 ? t.compW : t.rodW) / cropW; break;
        case Fit::FitHeight: sx = sy = (t.compH > 0 ? t.compH : t.rodH) / cropH; break;
        default: sx = sy = std::min(sx, sy); break;
    }
    dp.scaleX = static_cast<float>(sx);
    dp.scaleY = static_cast<float>(sy);
    dp.offX = static_cast<float>((t.rodW - cropW * sx) / 2);
    dp.offY = static_cast<float>((t.rodH - cropH * sy) / 2);
    dp.nearest = std::fabs(sx - 1) < 1e-6 && std::fabs(sy - 1) < 1e-6 &&
                 std::fabs(dp.offX - std::round(dp.offX)) < 1e-4 && std::fabs(dp.offY - std::round(dp.offY)) < 1e-4;
    if (dp.stab.on) dp.nearest = 0;
    dp.xform = 0;
    const Transform& x = s.xf;
    if (x.neutral()) return;
    // Forward: fitted frame -> output, both centred with y up, in render-scaled pixels:
    // flip, then about the anchor: zoom, rotation, pitch and yaw (a card seen in perspective
    // from a distance of one frame width), then the position.
    const double rx = s.renderScaleX, ry = s.renderScaleY, pi = 3.14159265358979323846;
    const double ax = x.anchorX * rx, ay = x.anchorY * ry;
    auto mul = [](const double a[9], const double b[9], double o[9]) {
        double r[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] + a[3 * i + 2] * b[6 + j];
        std::memcpy(o, r, sizeof r);
    };
    double m[9] = {x.flipH ? -1.0 : 1.0, 0, 0, 0, x.flipV ? -1.0 : 1.0, 0, 0, 0, 1};
    const double toAnchor[9] = {1, 0, -ax, 0, 1, -ay, 0, 0, 1};
    mul(toAnchor, m, m);
    const double zoom[9] = {x.zoomX, 0, 0, 0, x.zoomY, 0, 0, 0, 1};
    mul(zoom, m, m);
    const double a = x.rotation * pi / 180, c = std::cos(a), sn = std::sin(a);
    const double rot[9] = {c, -sn, 0, sn, c, 0, 0, 0, 1};
    mul(rot, m, m);
    if (x.pitch != 0 || x.yaw != 0) {
        // The plane z = 0 turned about x (pitch: positive leans the top away) and then about y
        // (yaw: positive turns the right side away), z pointing away from the viewer, seen from
        // z = -D and projected back onto z = 0.
        const double p = x.pitch * pi / 180, y = x.yaw * pi / 180, D = t.rodW;
        const double cp = std::cos(p), sp = std::sin(p), cy = std::cos(y), sy2 = std::sin(y);
        // columns: images of the x and y axes of the plane
        const double X0 = cy, Y0 = 0, Z0 = sy2;                    // (1,0,0) -> about y
        const double X1 = -sy2 * sp, Y1 = cp, Z1 = cy * sp;        // (0,1,0) -> about x, then y
        const double persp[9] = {D * X0, D * X1, 0, D * Y0, D * Y1, 0, Z0, Z1, D};
        mul(persp, m, m);
    }
    const double back[9] = {1, 0, ax + x.posX * rx, 0, 1, ay + x.posY * ry, 0, 0, 1};
    mul(back, m, m);
    // The kernel needs output -> fitted: the inverse.
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (std::fabs(det) < 1e-12) {   // zoom 0: nothing of the picture is left
        const double none[9] = {0, 0, 1e9, 0, 0, 1e9, 0, 0, 1};
        for (int i = 0; i < 9; ++i) dp.xf[i] = static_cast<float>(none[i]);
    } else {
        const double inv[9] = {(m[4] * m[8] - m[5] * m[7]) / det, (m[2] * m[7] - m[1] * m[8]) / det, (m[1] * m[5] - m[2] * m[4]) / det,
                               (m[5] * m[6] - m[3] * m[8]) / det, (m[0] * m[8] - m[2] * m[6]) / det, (m[2] * m[3] - m[0] * m[5]) / det,
                               (m[3] * m[7] - m[4] * m[6]) / det, (m[1] * m[6] - m[0] * m[7]) / det, (m[0] * m[4] - m[1] * m[3]) / det};
        // Keep w positive in front of the viewer.
        const double sgn = inv[8] < 0 ? -1.0 : 1.0;
        for (int i = 0; i < 9; ++i) dp.xf[i] = static_cast<float>(sgn * inv[i]);
    }
    dp.xform = 1;
    dp.nearest = 0;
}

bool Developer::develop(const Frame& f, const RawSettings& s, CUstream stream, const Target& t,
                        std::string& e, DevelopTiming* timing, const SourceImage* source) {
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
    dp.stab = s.stab;
    if (const LensWarp* w = s.lensDistortion && s.lensProfile.valid ? &s.lensProfile : nullptr) {
        // The profile's radius is in pixels of the whole sensor (6048 x 4032); a frame from a
        // smaller sensor window shows its middle, at the frame's own scale.
        const double scale = info.cropW / (6048.0 * std::clamp(f.profileW, 0.05, 1.0));   // frame pixels per sensor pixel
        const double m = (w->radius > 0 ? w->radius : 0.5 * std::sqrt(6048.0 * 6048.0 + 4032.0 * 4032.0)) * scale;
        dp.lens.on = 1;
        dp.lens.cx = static_cast<float>(info.activeLeft + info.cropX + w->cx * info.cropW);
        dp.lens.cy = static_cast<float>(info.activeTop + info.cropY + w->cy * info.cropH);
        dp.lens.m = static_cast<float>(m);
        dp.lens.invM = static_cast<float>(1.0 / m);
        for (int i = 0; i < 7; ++i) dp.lens.c[i] = static_cast<float>(w->c[i]);
    }
    if (source) {
        // The planes are the host's picture: the timeline's frame with the clip's frame placed in
        // it by the host's input scaling, centred, at one scale k: the largest that shows it whole
        // (scale to fit, bars beside it) or the smallest that fills the frame (fill, overhang cut).
        // Everything measured in raster pixels moves to that picture's pixels.
        if (source->width <= 0 || source->height <= 0 || source->rodW <= 0 || source->rodH <= 0 || (!source->device && !source->host)) { e = "Bad source image"; return false; }
        const double fitK = std::min(static_cast<double>(source->width) / info.cropW, static_cast<double>(source->height) / info.cropH);
        const double fillK = std::max(static_cast<double>(source->width) / info.cropW, static_cast<double>(source->height) / info.cropH);
        const double k = s.sourceFill ? fillK : fitK;
        const double picW = info.cropW * k, picH = info.cropH * k;
        const double px0 = (source->width - picW) / 2, py0 = (source->height - picH) / 2;
        const double rcx = info.activeLeft + info.cropX, rcy = info.activeTop + info.cropY;
        dp.W = source->width; dp.H = source->height;
        // Nothing to move (no stabilisation, no lens correction, the Transform and the Fit at
        // their defaults): the host's picture goes through as it is, wherever the clip sits in
        // it. Otherwise only the clip's frame is picture; what lies beside it stays black.
        const bool untouched = !dp.stab.on && !dp.lens.on && !s.sourceVignette.valid() && s.xf.neutral() && s.fit == Fit::Fit;
        if (untouched) {
            dp.cropX = dp.cropY = 0;
            dp.cropW = dp.W; dp.cropH = dp.H;
        } else {
            dp.cropX = std::clamp(static_cast<int>(std::lround(px0)), 0, dp.W - 1);
            dp.cropY = std::clamp(static_cast<int>(std::lround(py0)), 0, dp.H - 1);
            dp.cropW = std::clamp(static_cast<int>(std::lround(picW)), 1, dp.W - dp.cropX);
            dp.cropH = std::clamp(static_cast<int>(std::lround(picH)), 1, dp.H - dp.cropY);
        }
        if (dp.stab.on) {
            dp.stab.focal *= static_cast<float>(k);
            dp.stab.cx = static_cast<float>(px0 + (dp.stab.cx - rcx) * k);
            dp.stab.cy = static_cast<float>(py0 + (dp.stab.cy - rcy) * k);
            dp.stab.rows *= static_cast<float>(k);
            dp.stab.row0 = static_cast<float>(py0 - rcy * k);
        }
        if (dp.lens.on) {
            dp.lens.cx = static_cast<float>(px0 + (dp.lens.cx - rcx) * k);
            dp.lens.cy = static_cast<float>(py0 + (dp.lens.cy - rcy) * k);
            dp.lens.m *= static_cast<float>(k);
            dp.lens.invM = 1.f / dp.lens.m;
        }
        dp.passthrough = 1;
        if (s.sourceVignette.valid()) {
            // The sensor's centre is the picture's centre; its half diagonal in this picture's pixels.
            const double perSensorPx = k * info.cropW / (6048.0 * std::clamp(f.profileW, 0.05, 1.0));
            dp.vig.on = 1;
            dp.vig.cx = static_cast<float>(px0 + picW / 2);
            dp.vig.cy = static_cast<float>(py0 + picH / 2);
            dp.vig.invR = static_cast<float>(1.0 / (0.5 * std::sqrt(6048.0 * 6048.0 + 4032.0 * 4032.0) * perSensorPx));
            fit_vignette(s.sourceVignette, dp.vig.a);
            dp.vig.gamma = std::clamp(s.sourceGamma, 0, 10);
            dp.nearest = 0;
        }
    }
    frame_geometry(dp.cropW, dp.cropH, s, t, dp);
    if (dp.lens.on) dp.nearest = 0;
    if (!source) {
        dp.cropX = info.activeLeft + info.cropX;
        dp.cropY = info.activeTop + info.cropY;
    }
    resample_scale(s, dp);
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
    static const bool envCpu = os::env("SFP_CPU") == "1";
    const bool forceCpu = envCpu || gForceCpu;
    auto& cu = cuda();
    const void* srcHost = source ? source->host : nullptr;
    const int srcRow = source ? source->rowBytes : 0;
    if (!t.device && (forceCpu || !cu.ok)) {
        dp.rowBytes = t.rowBytes;
        // A Mac: its GPU through Metal. Should that fail, the processor does this frame.
        if (!forceCpu && metal_available()) {
            std::string why;
            if (develop_metal(f, s.decodeQuality, pp, dz, dp, t.host, why, timing, srcHost, srcRow)) return true;
        }
        return develop_cpu(f, s.decodeQuality, pp, dz, dp, t.host, e, timing, srcHost, srcRow);
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
            return develop_cpu(f, s.decodeQuality, pp, dz, dp, t.host, e, timing, srcHost, srcRow);
        }
        if (source) { e = "This GPU cannot run the plug-in's kernels: switch Develop RAW on"; return false; }
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
    const size_t n = source ? static_cast<size_t>(dp.W) * dp.H : static_cast<size_t>(W) * H;
    if (source && x->cap < n) {
        // The host's own picture: three planes are all it takes (a quarter of a gigabyte less
        // GPU memory at UHD than the buffers of a development).
        if (x->planeCap < n) {
            for (CUdeviceptr* p : {&x->raw, &x->raw2, &x->cfa, &x->vhb, &x->lpfb, &x->pqb, &x->R, &x->G, &x->B, &x->T[0], &x->T[1], &x->T[2], &x->T[3], &x->T[4], &x->T[5]})
                if (*p) { cu.cuMemFree(*p); *p = 0; }
            x->cap = x->planeCap = 0;
            x->valid = false;
            for (CUdeviceptr* p : {&x->R, &x->G, &x->B})
                if (!check(cu.cuMemAlloc(p, n * 4), "GPU memory", e)) return false;
            x->planeCap = n;
        }
    } else if (x->cap < n) {
        for (CUdeviceptr* p : {&x->raw, &x->raw2, &x->cfa, &x->vhb, &x->lpfb, &x->pqb, &x->R, &x->G, &x->B, &x->T[0], &x->T[1], &x->T[2], &x->T[3], &x->T[4], &x->T[5]})
            if (*p) { cu.cuMemFree(*p); *p = 0; }
        x->cap = x->planeCap = 0;
        x->valid = false;
        if (!check(cu.cuMemAlloc(&x->raw, n * 2), "GPU memory", e) || !check(cu.cuMemAlloc(&x->raw2, n * 2), "GPU memory", e)) return false;
        for (CUdeviceptr* p : {&x->cfa, &x->vhb, &x->lpfb, &x->pqb, &x->R, &x->G, &x->B, &x->T[0], &x->T[1], &x->T[2], &x->T[3], &x->T[4], &x->T[5]})
            if (!check(cu.cuMemAlloc(p, n * 4), "GPU memory", e)) return false;
        x->cap = x->planeCap = n;
    }
    auto t0 = std::chrono::steady_clock::now();
    const unsigned bx = 32, by = 8;
    auto launch = [&](CUfunction fn, unsigned w, unsigned h, void** args, const char* name) {
        return check(cu.cuLaunchKernel(fn, (w + bx - 1) / bx, (h + by - 1) / by, 1, bx, by, 1, 0, stream, args, nullptr), name, e);
    };
    bool reuse = !source && x->valid && x->frameId == f.id && x->quality == s.decodeQuality && x->lastDezig == dz && !std::memcmp(&x->last, &pp, sizeof pp);
    if (source) {
        x->valid = false;
        CUdeviceptr in = source->device;
        int rowBytes = source->rowBytes;
        if (!in) {
            // A host image: its rows packed, then to the GPU.
            const size_t packed = static_cast<size_t>(dp.W) * 16, bytes = packed * dp.H;
            if (x->srcCap < bytes) {
                if (x->src) cu.cuMemFree(x->src);
                x->src = 0;
                x->srcCap = 0;
                if (!check(cu.cuMemAlloc(&x->src, bytes), "GPU memory", e)) return false;
                x->srcCap = bytes;
            }
            x->staging.resize(bytes);
            for (int y = 0; y < dp.H; ++y)
                std::memcpy(x->staging.data() + y * packed, static_cast<const char*>(source->host) + static_cast<long long>(y) * source->rowBytes, packed);
            // The staging memory is used again for the download: have the upload finished first.
            if (!check(cu.cuMemcpyHtoDAsync(x->src, x->staging.data(), bytes, stream), "Upload source image", e) ||
                !check(cu.cuStreamSynchronize(stream), "Upload source image", e)) return false;
            in = x->src;
            rowBytes = static_cast<int>(packed);
        }
        void* a[] = {&in, &x->R, &x->G, &x->B, &dp.W, &dp.H, &rowBytes};
        if (!launch(x->unpack, dp.W, dp.H, a, "k_unpack")) return false;
        x->planes[0] = x->R; x->planes[1] = x->G; x->planes[2] = x->B;
    } else if (!reuse) {
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
