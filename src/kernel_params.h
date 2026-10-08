// Parameter blocks shared by the host and the CUDA kernels (4-byte fields only).
#pragma once

struct PrepParams {
    int W, H;
    int cfa[4];       // colour at (y&1)*2+(x&1): 0 R, 1 G, 2 B
    float black[4];
    float scale[4];   // wb[c] / (white - black) / norm
    float phase;      // row phase (plane pitch); 0 = off
    float white;
    int hotPixels;    // suppress isolated same-colour outliers
    float pedestal;   // added before demosaic so no noise is clipped at zero
};

// Gyro stabilisation warp (see gyro.h). A pixel of the stabilised (virtual, global-shutter)
// image is turned into a ray, rotated into the real camera at the time its sensor row was
// read, and projected again (pinhole). rot[k] is that rotation at knot k of the readout:
// knot 0 = first raster row, knot STAB_KNOTS-1 = last raster row.
#define STAB_KNOTS 33

struct StabParams {
    int on;               // 0 = no warp
    float focal;          // pixels at raster scale
    float cx, cy;         // principal point, raster coordinates
    float rows;           // raster rows covered by the readout
    float invZoom;        // 1 / (auto zoom * manual zoom)
    float rot[STAB_KNOTS * 9];
    float row0;           // raster row the readout starts at (0, except for a picture placed inside a larger image)
};

#ifdef __CUDACC__
#define SFP_FN __device__ __forceinline__
#else
#include <cmath>
#define SFP_FN inline
#endif

// Raster position (x, y) of the stabilised image -> raster position in the recorded frame.
// The row that was read at the matching time is found by fixed-point iteration (the
// vertical correction is far smaller than the frame, so three rounds are enough).
SFP_FN void stab_map(const StabParams& s, float x, float y, float* ox, float* oy) {
    float px = (x - s.cx) * s.invZoom, py = (y - s.cy) * s.invZoom, pz = s.focal;
    float sx = x, sy = y;
    for (int it = 0; it < 3; ++it) {
        float kf = (sy - s.row0) / s.rows * (float)(STAB_KNOTS - 1);
        kf = kf < 0.f ? 0.f : (kf > (float)(STAB_KNOTS - 1) ? (float)(STAB_KNOTS - 1) : kf);
        int k = (int)kf;
        if (k > STAB_KNOTS - 2) k = STAB_KNOTS - 2;
        float t = kf - (float)k;
        const float* a = s.rot + 9 * k;
        const float* b = a + 9;
        float m[9];
        for (int i = 0; i < 9; ++i) m[i] = a[i] + t * (b[i] - a[i]);
        float qx = m[0] * px + m[1] * py + m[2] * pz;
        float qy = m[3] * px + m[4] * py + m[5] * pz;
        float qz = m[6] * px + m[7] * py + m[8] * pz;
        if (qz < 1e-3f * s.focal) { sx = -1e9f; sy = -1e9f; break; }
        sx = s.cx + s.focal * qx / qz;
        sy = s.cy + s.focal * qy / qz;
    }
    *ox = sx; *oy = sy;
}

// Lens distortion (the DNG's WarpRectilinear opcode, as the camera writes it from the lens's own
// data): a position of the corrected picture -> where the lens put it in the recorded frame.
struct LensParams {
    int on;           // 0 = no correction
    float cx, cy;     // optical centre, raster coordinates
    float m, invM;    // normalisation radius (centre to the farthest corner of the area the profile is for), raster pixels
    float c[7];       // radial factor: sum of c[i] r^i
};

SFP_FN void lens_map(const LensParams& l, float x, float y, float* ox, float* oy) {
    const float dx = (x - l.cx) * l.invM, dy = (y - l.cy) * l.invM;
    const float r = sqrtf(dx * dx + dy * dy);
    const float f = l.c[0] + r * (l.c[1] + r * (l.c[2] + r * (l.c[3] + r * (l.c[4] + r * (l.c[5] + r * l.c[6])))));
    *ox = l.cx + dx * f * l.m;
    *oy = l.cy + dy * f * l.m;
}

// Vignette correction of the host's own picture (Develop RAW off): a radial brightness gain
// 1 + sum a[n] r^(2n+2), applied in linear light; the picture is taken out of and put back into
// its encoding (gamma: the ids of pt_decode in kernels.cu).
struct VignetteParams {
    int on;
    float cx, cy;     // the sensor's centre in the picture, plane pixels
    float invR;       // 1 / the sensor's half diagonal in plane pixels
    float a[4];
    int gamma;
};

struct DevelopParams {
    int W, H;                         // raw dimensions
    int cropX, cropY, cropW, cropH;   // DefaultCrop in raw coordinates
    int outW, outH, rowBytes;         // output image (bounds) and its row pitch in bytes
    int boundsX, boundsY, rodH;       // bounds origin within the region of definition
    float scaleX, scaleY, offX, offY; // RoD pixel = crop pixel * scale + off
    int nearest;
    float sharpen, norm, clip;
    float chanClip[3];
    int highlightRecovery;
    float m[9];
    float gainLin;
    float lum[3];
    int toneActive;
    float contrast, midtones, highlights, shadows;
    float colorBoost, saturation;
    int gamutMap, preToneCurve, softClip, gamma;
    float lift, gain;
    float pedestal;   // removed after sampling the demosaiced planes
    StabParams stab;  // gyro stabilisation (stab.on == 0: none)
    int rodW;         // region of definition width (the timeline frame at render scale)
    int xform;        // 1: the Transform controls are not neutral, xf is used
    float xf[9];      // output -> fitted frame, both centred with y up (homogeneous 3x3, row major)
    int resampler;    // 0 bilinear, 1 Catmull-Rom, 2 Lanczos-2, 3 Lanczos-3
    float kx, ky;     // source pixels per output pixel along x / y (>= 1): the kernel is widened by it
    LensParams lens;  // lens distortion correction (lens.on == 0: none), after the stabilisation's map
    int passthrough;  // 1: the planes hold the host's own picture; it is resampled and written as it is (no development)
    VignetteParams vig;   // pass-through only
};

enum { kResampleBilinear = 0, kResampleCatmullRom = 1, kResampleLanczos2 = 2, kResampleLanczos3 = 3 };

// The Transform controls: from an output position (RoD pixel coordinates, y down, pixel centres
// at +0.5) to the position in the fitted (untransformed) frame. False when the point lies
// behind the viewer (pitch / yaw beyond 90 degrees). One mapping for the GPU, the processor and
// the tests, so the picture is resampled once, wherever it is computed.
SFP_FN bool xform_map(const DevelopParams& p, float ix, float iy, float* ox, float* oy) {
    const float u = ix - 0.5f * (float)p.rodW, v = 0.5f * (float)p.rodH - iy;
    const float X = p.xf[0] * u + p.xf[1] * v + p.xf[2];
    const float Y = p.xf[3] * u + p.xf[4] * v + p.xf[5];
    const float W = p.xf[6] * u + p.xf[7] * v + p.xf[8];
    if (W <= 1e-6f) return false;
    *ox = X / W + 0.5f * (float)p.rodW;
    *oy = 0.5f * (float)p.rodH - Y / W;
    return true;
}

struct DezigParams {
    int W, H;
    float strength;    // 0..1 blend
    float sigma;       // along-edge Gaussian sigma (pixels)
    float rangeSigma;  // luma similarity (sqrt domain)
    float minSlope;    // edges steeper than this (|dy/dx| of the edge) are left alone
    float minEnergy;   // tensor trace below this is noise/flat
};
