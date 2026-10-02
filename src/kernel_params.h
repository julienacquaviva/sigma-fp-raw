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
        float kf = sy / s.rows * (float)(STAB_KNOTS - 1);
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
};

struct DezigParams {
    int W, H;
    float strength;    // 0..1 blend
    float sigma;       // along-edge Gaussian sigma (pixels)
    float rangeSigma;  // luma similarity (sqrt domain)
    float minSlope;    // edges steeper than this (|dy/dx| of the edge) are left alone
    float minEnergy;   // tensor trace below this is noise/flat
};
