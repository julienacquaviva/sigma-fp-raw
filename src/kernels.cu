// Sigma fp RAW CUDA kernels (compiled to PTX at build time with NVRTC; no CUDA
// headers). Pipeline: prep (linearise, white balance, row phase) -> RCD demosaic
// (VH, LPF, green, PQ, R/B at R/B, R/B at G) or half-res bin -> develop (sample, sharpen,
// camera matrix, RAW controls, output encoding) into the host's RGBA float image.
#include "kernel_params.h"

// The same source is the CUDA kernels (NVRTC) and, included by kernels_cpu.cpp, the CPU
// version: there a kernel is a plain function called once per pixel with its position.
#ifdef __CUDACC__
#define SFP_INL __device__ __forceinline__
#define SFP_DEV __device__
#define SFP_KERNEL extern "C" __global__ void
#define SFP_XY_ARGS
#define SFP_XY(vx, vy) int vx = blockIdx.x * blockDim.x + threadIdx.x, vy = blockIdx.y * blockDim.y + threadIdx.y
#define SFP_BYVAL(T) T
#define SFP_EXP __expf
#else
#define SFP_INL static inline
#define SFP_DEV static inline
#define SFP_KERNEL static inline void
#define SFP_XY_ARGS int sfpX, int sfpY,
#define SFP_XY(vx, vy) int vx = sfpX, vy = sfpY
#define SFP_BYVAL(T) const T&
#define SFP_EXP expf
#endif

#define EPS 1e-5f
#define EPSSQ 1e-10f

SFP_INL int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
SFP_INL float clampf(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }
SFP_INL float sqr(float v) { return v * v; }
SFP_INL float smooth(float e0, float e1, float x) {
    float t = clampf((x - e0) / (e1 - e0), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}
// Reflect about the first/last sample; reflection about an integer keeps CFA parity.
SFP_INL int mir(int v, int n) {
    if (v < 0) v = -v;
    if (v >= n) v = 2 * n - 2 - v;
    return clampi(v, 0, n - 1);
}
SFP_INL int fc(const PrepParams& p, int x, int y) { return p.cfa[((y & 1) << 1) | (x & 1)]; }

// Keys cubic (a = -0.5) weights for fractional t.
SFP_INL void cubic_w(float t, float w[4]) {
    const float a = -0.5f;
    float t2 = t * t, t3 = t2 * t;
    w[0] = a * t3 - 2.f * a * t2 + a * t;
    w[1] = (a + 2.f) * t3 - (a + 3.f) * t2 + 1.f;
    w[2] = -(a + 2.f) * t3 + (2.f * a + 3.f) * t2 - a * t;
    w[3] = -a * t3 + a * t2;
}

// Hot pixels: a sample brighter than all 8 same-colour neighbours by a noise-scaled
// margin is replaced by the brightest neighbour (a copy otherwise).
SFP_KERNEL k_hot(SFP_XY_ARGS const unsigned short* raw, unsigned short* out, SFP_BYVAL(PrepParams) p) {
    SFP_XY(x, y);
    if (x >= p.W || y >= p.H) return;
    size_t i = (size_t)y * p.W + x;
    unsigned short c = raw[i];
    if (x >= 2 && y >= 2 && x < p.W - 2 && y < p.H - 2) {
        int k = ((y & 1) << 1) | (x & 1);
        float mx = 0.f, s1 = 0.f, s2 = 0.f;
        for (int dy = -2; dy <= 2; dy += 2)
            for (int dx = -2; dx <= 2; dx += 2) {
                if (!dx && !dy) continue;
                float n = (float)raw[i + (long long)dy * p.W + dx];
                mx = fmaxf(mx, n); s1 += n; s2 += n * n;
            }
        float mean = s1 * 0.125f, sd = sqrtf(fmaxf(s2 * 0.125f - mean * mean, 0.f));
        if ((float)c > mx + fmaxf(4.f * sd, 0.012f * (p.white - p.black[k]))) c = (unsigned short)mx;
    }
    out[i] = c;
}

// raw uint16 -> normalised, white-balanced CFA with per-parity vertical row phase.
// Rows of parity 0 sample the same-colour plane at j + phase, parity 1 at j - phase
// (plane pitch units), cubic; phase 0 is a straight copy.
SFP_KERNEL k_prep(SFP_XY_ARGS const unsigned short* raw, float* cfa, SFP_BYVAL(PrepParams) p) {
    SFP_XY(x, y);
    if (x >= p.W || y >= p.H) return;
    int k = ((y & 1) << 1) | (x & 1);
    float v;
    if (p.phase == 0.f) {
        v = (float)raw[(size_t)y * p.W + x];
    } else {
        int par = y & 1, n = (p.H - par + 1) >> 1, j = y >> 1;
        float pos = (float)j + (par ? -p.phase : p.phase);
        int j0 = (int)floorf(pos);
        float w[4];
        cubic_w(pos - (float)j0, w);
        v = 0.f;
        for (int i = 0; i < 4; ++i) {
            int jj = clampi(j0 - 1 + i, 0, n - 1);
            v += w[i] * (float)raw[(size_t)(par + 2 * jj) * p.W + x];
        }
    }
    cfa[(size_t)y * p.W + x] = (v - p.black[k]) * p.scale[k] + p.pedestal;
}

#define CFA(xx, yy) cfa[(size_t)mir(yy, H) * W + mir(xx, W)]

// Step 1: vertical/horizontal discrimination VH_Dir = V/(V+H) of squared high-pass sums.
SFP_INL float hpfv(const float* cfa, int W, int H, int x, int y) {
    return sqr((CFA(x, y - 3) - CFA(x, y - 1) - CFA(x, y + 1) + CFA(x, y + 3)) - 3.f * (CFA(x, y - 2) + CFA(x, y + 2)) + 6.f * CFA(x, y));
}
SFP_INL float hpfh(const float* cfa, int W, int H, int x, int y) {
    return sqr((CFA(x - 3, y) - CFA(x - 1, y) - CFA(x + 1, y) + CFA(x + 3, y)) - 3.f * (CFA(x - 2, y) + CFA(x + 2, y)) + 6.f * CFA(x, y));
}
SFP_KERNEL k_vh(SFP_XY_ARGS const float* cfa, float* vh, int W, int H) {
    SFP_XY(x, y);
    if (x >= W || y >= H) return;
    float V = fmaxf(EPSSQ, hpfv(cfa, W, H, x, y - 1) + hpfv(cfa, W, H, x, y) + hpfv(cfa, W, H, x, y + 1));
    float Hs = fmaxf(EPSSQ, hpfh(cfa, W, H, x - 1, y) + hpfh(cfa, W, H, x, y) + hpfh(cfa, W, H, x + 1, y));
    vh[(size_t)y * W + x] = V / (V + Hs);
}

// Step 2: low-pass filter at R/B sites (stored full size; used only at R/B).
SFP_KERNEL k_lpf(SFP_XY_ARGS const float* cfa, float* lpf, SFP_BYVAL(PrepParams) p) {
    SFP_XY(x, y);
    int W = p.W, H = p.H;
    if (x >= W || y >= H) return;
    if (fc(p, x, y) == 1) { lpf[(size_t)y * W + x] = 0.f; return; }
    lpf[(size_t)y * W + x] = CFA(x, y) + 0.5f * (CFA(x, y - 1) + CFA(x, y + 1) + CFA(x - 1, y) + CFA(x + 1, y)) +
                             0.25f * (CFA(x - 1, y - 1) + CFA(x + 1, y - 1) + CFA(x - 1, y + 1) + CFA(x + 1, y + 1));
}

#define LPF(xx, yy) fmaxf(lpf[(size_t)mir(yy, H) * W + mir(xx, W)], 0.f)
#define VH(xx, yy) vh[(size_t)mir(yy, H) * W + mir(xx, W)]

// Step 3: green at R/B sites (copy at G sites).
SFP_KERNEL k_green(SFP_XY_ARGS const float* cfa, const float* vh, const float* lpf, float* G, SFP_BYVAL(PrepParams) p) {
    SFP_XY(x, y);
    int W = p.W, H = p.H;
    if (x >= W || y >= H) return;
    size_t i = (size_t)y * W + x;
    if (fc(p, x, y) == 1) { G[i] = cfa[i]; return; }
    float c = VH(x, y);
    float nb = 0.25f * (VH(x - 1, y - 1) + VH(x + 1, y - 1) + VH(x - 1, y + 1) + VH(x + 1, y + 1));
    float disc = fabsf(0.5f - c) < fabsf(0.5f - nb) ? nb : c;
    float c0 = CFA(x, y);
    float nG = EPS + fabsf(CFA(x, y - 1) - CFA(x, y + 1)) + fabsf(c0 - CFA(x, y - 2)) + fabsf(CFA(x, y - 1) - CFA(x, y - 3)) + fabsf(CFA(x, y - 2) - CFA(x, y - 4));
    float sG = EPS + fabsf(CFA(x, y - 1) - CFA(x, y + 1)) + fabsf(c0 - CFA(x, y + 2)) + fabsf(CFA(x, y + 1) - CFA(x, y + 3)) + fabsf(CFA(x, y + 2) - CFA(x, y + 4));
    float wG = EPS + fabsf(CFA(x - 1, y) - CFA(x + 1, y)) + fabsf(c0 - CFA(x - 2, y)) + fabsf(CFA(x - 1, y) - CFA(x - 3, y)) + fabsf(CFA(x - 2, y) - CFA(x - 4, y));
    float eG = EPS + fabsf(CFA(x - 1, y) - CFA(x + 1, y)) + fabsf(c0 - CFA(x + 2, y)) + fabsf(CFA(x + 1, y) - CFA(x + 3, y)) + fabsf(CFA(x + 2, y) - CFA(x + 4, y));
    float l0 = LPF(x, y);
    float nE = CFA(x, y - 1) * (1.f + (l0 - LPF(x, y - 2)) / (EPS + l0 + LPF(x, y - 2)));
    float sE = CFA(x, y + 1) * (1.f + (l0 - LPF(x, y + 2)) / (EPS + l0 + LPF(x, y + 2)));
    float wE = CFA(x - 1, y) * (1.f + (l0 - LPF(x - 2, y)) / (EPS + l0 + LPF(x - 2, y)));
    float eE = CFA(x + 1, y) * (1.f + (l0 - LPF(x + 2, y)) / (EPS + l0 + LPF(x + 2, y)));
    float vE = (sG * nE + nG * sE) / (nG + sG);
    float hE = (wG * eE + eG * wE) / (eG + wG);
    G[i] = clampf(disc * hE + (1.f - disc) * vE, 0.f, 2.f);
}

// Step 4.1: P/Q diagonal discrimination at R/B sites.
SFP_INL float hpfp(const float* cfa, int W, int H, int x, int y) {
    return sqr((CFA(x - 3, y - 3) - CFA(x - 1, y - 1) - CFA(x + 1, y + 1) + CFA(x + 3, y + 3)) - 3.f * (CFA(x - 2, y - 2) + CFA(x + 2, y + 2)) + 6.f * CFA(x, y));
}
SFP_INL float hpfq(const float* cfa, int W, int H, int x, int y) {
    return sqr((CFA(x + 3, y - 3) - CFA(x + 1, y - 1) - CFA(x - 1, y + 1) + CFA(x - 3, y + 3)) - 3.f * (CFA(x + 2, y - 2) + CFA(x - 2, y + 2)) + 6.f * CFA(x, y));
}
SFP_KERNEL k_pq(SFP_XY_ARGS const float* cfa, float* pq, SFP_BYVAL(PrepParams) p) {
    SFP_XY(x, y);
    int W = p.W, H = p.H;
    if (x >= W || y >= H) return;
    if (fc(p, x, y) == 1) { pq[(size_t)y * W + x] = 0.5f; return; }
    float P = fmaxf(EPSSQ, hpfp(cfa, W, H, x - 1, y - 1) + hpfp(cfa, W, H, x, y) + hpfp(cfa, W, H, x + 1, y + 1));
    float Q = fmaxf(EPSSQ, hpfq(cfa, W, H, x + 1, y - 1) + hpfq(cfa, W, H, x, y) + hpfq(cfa, W, H, x - 1, y + 1));
    pq[(size_t)y * W + x] = P / (P + Q);
}

#define PQ(xx, yy) pq[(size_t)mir(yy, H) * W + mir(xx, W)]
#define PL(pl, xx, yy) pl[(size_t)mir(yy, H) * W + mir(xx, W)]

// Step 4.2: the missing R or B at B/R sites (diagonal neighbours carry it natively).
// Planes R, B hold cfa at their own sites after this kernel.
SFP_KERNEL k_rb_diag(SFP_XY_ARGS const float* cfa, const float* G, const float* pq, float* R, float* B, SFP_BYVAL(PrepParams) p) {
    SFP_XY(x, y);
    int W = p.W, H = p.H;
    if (x >= W || y >= H) return;
    size_t i = (size_t)y * W + x;
    int col = fc(p, x, y);
    if (col == 1) return;
    float* own = col == 0 ? R : B;
    float* oth = col == 0 ? B : R;
    own[i] = cfa[i];
    // The other colour sits on the diagonal neighbours (raw values).
    float c = PQ(x, y);
    float nb = 0.25f * (PQ(x - 1, y - 1) + PQ(x + 1, y - 1) + PQ(x - 1, y + 1) + PQ(x + 1, y + 1));
    float disc = fabsf(0.5f - c) < fabsf(0.5f - nb) ? nb : c;
    float g0 = G[i];
    float cNW = CFA(x - 1, y - 1), cNE = CFA(x + 1, y - 1), cSW = CFA(x - 1, y + 1), cSE = CFA(x + 1, y + 1);
    float nwG = EPS + fabsf(cNW - cSE) + fabsf(cNW - CFA(x - 3, y - 3)) + fabsf(g0 - PL(G, x - 2, y - 2));
    float neG = EPS + fabsf(cNE - cSW) + fabsf(cNE - CFA(x + 3, y - 3)) + fabsf(g0 - PL(G, x + 2, y - 2));
    float swG = EPS + fabsf(cNE - cSW) + fabsf(cSW - CFA(x - 3, y + 3)) + fabsf(g0 - PL(G, x - 2, y + 2));
    float seG = EPS + fabsf(cNW - cSE) + fabsf(cSE - CFA(x + 3, y + 3)) + fabsf(g0 - PL(G, x + 2, y + 2));
    float nwE = cNW - PL(G, x - 1, y - 1), neE = cNE - PL(G, x + 1, y - 1);
    float swE = cSW - PL(G, x - 1, y + 1), seE = cSE - PL(G, x + 1, y + 1);
    float pE = (nwG * seE + seG * nwE) / (nwG + seG);
    float qE = (neG * swE + swG * neE) / (neG + swG);
    oth[i] = clampf(g0 + (1.f - disc) * pE + disc * qE, 0.f, 2.f);
}

// Step 4.3: R and B at G sites (N/S/W/E neighbours are R/B sites, complete after 4.2).
SFP_KERNEL k_rb_g(SFP_XY_ARGS const float* G, const float* vh, float* R, float* B, SFP_BYVAL(PrepParams) p) {
    SFP_XY(x, y);
    int W = p.W, H = p.H;
    if (x >= W || y >= H) return;
    if (fc(p, x, y) != 1) return;
    size_t i = (size_t)y * W + x;
    float c = VH(x, y);
    float nb = 0.25f * (VH(x - 1, y - 1) + VH(x + 1, y - 1) + VH(x - 1, y + 1) + VH(x + 1, y + 1));
    float disc = fabsf(0.5f - c) < fabsf(0.5f - nb) ? nb : c;
    float g0 = G[i];
    float gN = PL(G, x, y - 1), gS = PL(G, x, y + 1), gW = PL(G, x - 1, y), gE = PL(G, x + 1, y);
    for (int k = 0; k < 2; ++k) {
        float* pl = k ? B : R;
        float n = PL(pl, x, y - 1), s = PL(pl, x, y + 1), w = PL(pl, x - 1, y), e = PL(pl, x + 1, y);
        float nG = EPS + fabsf(g0 - PL(G, x, y - 2)) + fabsf(n - s) + fabsf(n - PL(pl, x, y - 3));
        float sG = EPS + fabsf(g0 - PL(G, x, y + 2)) + fabsf(s - n) + fabsf(s - PL(pl, x, y + 3));
        float wG = EPS + fabsf(g0 - PL(G, x - 2, y)) + fabsf(w - e) + fabsf(w - PL(pl, x - 3, y));
        float eG = EPS + fabsf(g0 - PL(G, x + 2, y)) + fabsf(e - w) + fabsf(e - PL(pl, x + 3, y));
        float nE = n - gN, sE = s - gS, wE = w - gW, eE = e - gE;
        float vE = (nG * sE + sG * nE) / (nG + sG);
        float hE = (eG * wE + wG * eE) / (eG + wG);
        pl[i] = clampf(g0 + (1.f - disc) * vE + disc * hE, 0.f, 2.f);
    }
}

// Half-resolution decode: each 2x2 cell -> one RGB sample at the cell centre,
// written to every pixel of the cell (develop samples at full-res coordinates).
SFP_KERNEL k_half(SFP_XY_ARGS const float* cfa, float* R, float* G, float* B, SFP_BYVAL(PrepParams) p) {
    SFP_XY(cx, cy);
    int W = p.W, H = p.H;
    if (2 * cx + 1 >= W || 2 * cy + 1 >= H) return;
    float s[3] = {0, 0, 0}, n[3] = {0, 0, 0};
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            int x = 2 * cx + dx, y = 2 * cy + dy, c = fc(p, x, y);
            s[c] += cfa[(size_t)y * W + x];
            n[c] += 1.f;
        }
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            size_t i = (size_t)(2 * cy + dy) * W + 2 * cx + dx;
            R[i] = s[0] / n[0]; G[i] = s[1] / n[1]; B[i] = s[2] / n[2];
        }
}

// De-zigzag: edge-directed smoothing along near-horizontal edges. The /2 binned
// readout (M98) aliases thin near-horizontal detail into a 2-row stair-step that no
// demosaic can undo; averaging along the local edge direction (structure tensor)
// removes the steps without blurring across the edge (luma range weight).
SFP_INL float lumaAt(const float* R, const float* G, const float* B, int W, int H, int x, int y) {
    size_t i = (size_t)clampi(y, 0, H - 1) * W + clampi(x, 0, W - 1);
    return sqrtf(fmaxf(0.25f * R[i] + 0.5f * G[i] + 0.25f * B[i], 0.f));
}
SFP_DEV void bilerp3(const float* R, const float* G, const float* B, int W, int H, float x, float y, float o[3]) {
    int x0 = (int)floorf(x), y0 = (int)floorf(y);
    float tx = x - x0, ty = y - y0;
    int x1 = clampi(x0 + 1, 0, W - 1), y1 = clampi(y0 + 1, 0, H - 1);
    x0 = clampi(x0, 0, W - 1); y0 = clampi(y0, 0, H - 1);
    size_t a = (size_t)y0 * W + x0, b = (size_t)y0 * W + x1, c = (size_t)y1 * W + x0, d = (size_t)y1 * W + x1;
    float w00 = (1 - tx) * (1 - ty), w01 = tx * (1 - ty), w10 = (1 - tx) * ty, w11 = tx * ty;
    o[0] = R[a] * w00 + R[b] * w01 + R[c] * w10 + R[d] * w11;
    o[1] = G[a] * w00 + G[b] * w01 + G[c] * w10 + G[d] * w11;
    o[2] = B[a] * w00 + B[b] * w01 + B[c] * w10 + B[d] * w11;
}
// Structure tensor of sqrt-luma (central differences), smoothed separably afterwards.
SFP_KERNEL k_tensor(SFP_XY_ARGS const float* R, const float* G, const float* B, float* Jxx, float* Jyy, float* Jxy, int W, int H) {
    SFP_XY(x, y);
    if (x >= W || y >= H) return;
    float gx = 0.5f * (lumaAt(R, G, B, W, H, x + 1, y) - lumaAt(R, G, B, W, H, x - 1, y));
    float gy = 0.5f * (lumaAt(R, G, B, W, H, x, y + 1) - lumaAt(R, G, B, W, H, x, y - 1));
    size_t i = (size_t)y * W + x;
    Jxx[i] = gx * gx; Jyy[i] = gy * gy; Jxy[i] = gx * gy;
}
// Separable Gaussian (sigma 2.5, radius 7) of three planes; dir 0 horizontal, 1 vertical.
SFP_KERNEL k_blur3(SFP_XY_ARGS const float* a, const float* b, const float* c, float* oa, float* ob, float* oc, int W, int H, int dir) {
    SFP_XY(x, y);
    if (x >= W || y >= H) return;
    float sa = 0.f, sb = 0.f, sc = 0.f, sw = 0.f;
    for (int t = -4; t <= 4; ++t) {
        int xx = dir ? x : clampi(x + t, 0, W - 1), yy = dir ? clampi(y + t, 0, H - 1) : y;
        size_t j = (size_t)yy * W + xx;
        float w = SFP_EXP(-(float)(t * t) / 4.5f);
        sa += w * a[j]; sb += w * b[j]; sc += w * c[j]; sw += w;
    }
    size_t i = (size_t)y * W + x;
    oa[i] = sa / sw; ob[i] = sb / sw; oc[i] = sc / sw;
}

// De-zigzag: smooth along sharp, straight, near-horizontal edges only. The /2 binned
// readout (M98) turns such edges into a 2-row stair-step with period 2/slope along the
// edge; averaging over about that period removes it. Soft/defocused or curved edges
// carry no stair-step and are left alone (user test: smearing on a defocused statue).
SFP_KERNEL k_dezig(SFP_XY_ARGS const float* R, const float* G, const float* B, const float* Jxx, const float* Jyy,
                                   const float* Jxy, float* R2, float* G2, float* B2, SFP_BYVAL(DezigParams) p) {
    SFP_XY(x, y);
    int W = p.W, H = p.H;
    if (x >= W || y >= H) return;
    size_t i = (size_t)y * W + x;
    float o0 = R[i], o1 = G[i], o2 = B[i];
    R2[i] = o0; G2[i] = o1; B2[i] = o2;
    float jxx = Jxx[i], jyy = Jyy[i], jxy = Jxy[i];
    float tr = jxx + jyy;
    if (jyy <= jxx || tr < p.minEnergy) return;                  // gradient must be mostly vertical
    float det = sqrtf(fmaxf(0.f, sqr(jxx - jyy) + 4.f * jxy * jxy));
    float coh = det / tr;                                          // 1 = perfectly straight edge
    float gate = smooth(0.4f, 0.75f, coh);
    if (gate <= 0.f) return;
    // Edge direction: eigenvector of the smaller eigenvalue (second row of (J - l_min I) v = 0).
    float ex = jxx - jyy - det, ey = 2.f * jxy;
    float len = sqrtf(ex * ex + ey * ey);
    if (len < 1e-12f) return;
    ex /= len; ey /= len;
    if (ex < 0.f) { ex = -ex; ey = -ey; }
    float slope = fabsf(ey) / fmaxf(ex, 1e-6f);
    gate *= 1.f - smooth(0.25f, p.minSlope, slope);
    if (gate <= 0.f) return;
    // Sharpness: a step (1st difference) or a thin line (2nd difference) must be as strong
    // at 1-row scale as at 3-row scale, at this row or a neighbour. Defocused edges fail.
    float sharp = 0.f;
    for (int d = -1; d <= 1; ++d) {
        int yy = y + d;
        float c = lumaAt(R, G, B, W, H, x, yy);
        float u1 = lumaAt(R, G, B, W, H, x, yy - 1), d1 = lumaAt(R, G, B, W, H, x, yy + 1);
        float u3 = lumaAt(R, G, B, W, H, x, yy - 3), d3 = lumaAt(R, G, B, W, H, x, yy + 3);
        float fine = fmaxf(fabsf(d1 - u1), fabsf(2.f * c - u1 - d1));
        float coarse = fmaxf(fabsf(d3 - u3), fabsf(2.f * c - u3 - d3));
        sharp = fmaxf(sharp, fine / (coarse + 1e-4f));
    }
    gate *= smooth(0.5f, 0.8f, sharp);
    if (gate <= 0.f) return;
    // Averaging length follows the stair-step period 2/slope (bounded).
    // Averaging length follows the stair-step period 2/slope, scaled by the amount (<= 12 px).
    float sigma = fminf(p.sigma, fmaxf(1.5f, 1.2f * p.strength / fmaxf(slope, 1e-3f)));
    float y0l = lumaAt(R, G, B, W, H, x, y);
    float acc[3] = {o0, o1, o2}, wsum = 1.f;
    const float inv2s2 = 1.f / (2.f * sigma * sigma), inv2r2 = 1.f / (2.f * p.rangeSigma * p.rangeSigma);
    int reach = (int)ceilf(2.5f * sigma);
    for (int t = -reach; t <= reach; ++t) {
        if (!t) continue;
        float s3[3];
        bilerp3(R, G, B, W, H, x + t * ex, y + t * ey, s3);
        float l = sqrtf(fmaxf(0.25f * s3[0] + 0.5f * s3[1] + 0.25f * s3[2], 0.f));
        float w = SFP_EXP(-(float)(t * t) * inv2s2 - sqr(l - y0l) * inv2r2);
        acc[0] += w * s3[0]; acc[1] += w * s3[1]; acc[2] += w * s3[2]; wsum += w;
    }
    float a = gate;
    R2[i] = o0 + a * (acc[0] / wsum - o0);
    G2[i] = o1 + a * (acc[1] / wsum - o1);
    B2[i] = o2 + a * (acc[2] / wsum - o2);
}

// ---- develop -------------------------------------------------------------------

SFP_DEV float encode(float v, int gamma) {
    switch (gamma) {
        case 1: return v > 0.f ? powf(v, 1.f / 2.2f) : v;
        case 2: return v > 0.f ? powf(v, 1.f / 2.4f) : v;
        case 3: return v < 0.018f ? 4.5f * v : 1.099f * powf(v, 0.45f) - 0.099f;          // Rec.709 OETF
        case 4: return v <= 0.0031308f ? 12.92f * v : 1.055f * powf(v, 1.f / 2.4f) - 0.055f;
        case 5: {                                                                            // DaVinci Intermediate
            const float A = 0.0075f, B = 7.0f, C = 0.07329248f, M = 10.44426855f, CUT = 0.00262409f;
            return v <= CUT ? v * M : (log2f(v + A) + B) * C;
        }
        case 6:                                                                              // ACEScct
            return v <= 0.0078125f ? 10.5402377416545f * v + 0.0729055341958355f : (log2f(v) + 9.72f) / 17.52f;
        default: return v;
    }
}
SFP_DEV float decode_g24(float v) { return v > 0.f ? powf(v, 2.4f) : v; }


SFP_DEV void sample3(const float* R, const float* G, const float* B, int W, int H, float sx, float sy, bool nearest, float o[3]) {
    if (nearest) {
        int x = clampi((int)floorf(sx), 0, W - 1), y = clampi((int)floorf(sy), 0, H - 1);
        size_t i = (size_t)y * W + x;
        o[0] = R[i]; o[1] = G[i]; o[2] = B[i];
        return;
    }
    float fx = sx - 0.5f, fy = sy - 0.5f;
    int x0 = (int)floorf(fx), y0 = (int)floorf(fy);
    float tx = fx - x0, ty = fy - y0;
    int x1 = clampi(x0 + 1, 0, W - 1), y1 = clampi(y0 + 1, 0, H - 1);
    x0 = clampi(x0, 0, W - 1); y0 = clampi(y0, 0, H - 1);
    size_t a = (size_t)y0 * W + x0, b = (size_t)y0 * W + x1, c = (size_t)y1 * W + x0, d = (size_t)y1 * W + x1;
    float w00 = (1 - tx) * (1 - ty), w01 = tx * (1 - ty), w10 = (1 - tx) * ty, w11 = tx * ty;
    o[0] = R[a] * w00 + R[b] * w01 + R[c] * w10 + R[d] * w11;
    o[1] = G[a] * w00 + G[b] * w01 + G[c] * w10 + G[d] * w11;
    o[2] = B[a] * w00 + B[b] * w01 + B[c] * w10 + B[d] * w11;
}

SFP_KERNEL k_develop(SFP_XY_ARGS const float* R, const float* G, const float* B, float* out, SFP_BYVAL(DevelopParams) p) {
    SFP_XY(ox, oy);
    if (ox >= p.outW || oy >= p.outH) return;
    float* o = (float*)((char*)out + (long long)oy * p.rowBytes) + 4 * ox;
    // Output pixel (ox, oy) is RoD pixel (boundsX + ox, boundsY + oy); OFX y is
    // bottom-up, so it shows image row (rodH - 1 - y) of the fitted frame.
    float ix = (float)(p.boundsX + ox) + 0.5f, iy = (float)(p.rodH - 1 - (p.boundsY + oy)) + 0.5f;
    float sx = (ix - p.offX) / p.scaleX, sy = (iy - p.offY) / p.scaleY;
    if (sx < 0.f || sy < 0.f || sx >= (float)p.cropW || sy >= (float)p.cropH) {
        o[0] = o[1] = o[2] = 0.f; o[3] = 1.f;
        return;
    }
    sx += (float)p.cropX; sy += (float)p.cropY;
    if (p.stab.on) {
        // Gyro stabilisation + rolling-shutter correction: where this pixel was recorded.
        stab_map(p.stab, sx, sy, &sx, &sy);
        if (sx < (float)p.cropX || sy < (float)p.cropY || sx >= (float)(p.cropX + p.cropW) || sy >= (float)(p.cropY + p.cropH)) {
            o[0] = o[1] = o[2] = 0.f; o[3] = 1.f;
            return;
        }
    }
    float c[3];
    sample3(R, G, B, p.W, p.H, sx, sy, p.nearest, c);
    if (p.sharpen > 0.f) {
        // Unsharp mask on the demosaiced signal (3x3 binomial blur at source pitch).
        float acc[3] = {0, 0, 0};
        const float k[3] = {0.25f, 0.5f, 0.25f};
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                float s[3];
                sample3(R, G, B, p.W, p.H, sx + dx, sy + dy, p.nearest, s);
                float w = k[dx + 1] * k[dy + 1];
                acc[0] += w * s[0]; acc[1] += w * s[1]; acc[2] += w * s[2];
            }
        for (int i = 0; i < 3; ++i) c[i] += p.sharpen * (c[i] - acc[i]);
    }
    // Remove the pedestal; back to white-balanced camera scale (sensor clip of green = 1).
    for (int i = 0; i < 3; ++i) c[i] = (c[i] - p.pedestal) * p.norm;
    // Highlights: clipped channels.
    float cmax = fmaxf(c[0], fmaxf(c[1], c[2]));
    if (!p.highlightRecovery) {
        for (int i = 0; i < 3; ++i) c[i] = fminf(c[i], p.clip);
    } else {
        // Blend towards the brightest channel as any channel nears its clip, keeping
        // the extra range the unclipped channels carry.
        float r = fmaxf(c[0] / p.chanClip[0], fmaxf(c[1] / p.chanClip[1], c[2] / p.chanClip[2]));
        float t = smooth(0.85f, 1.0f, r);
        for (int i = 0; i < 3; ++i) c[i] = c[i] + t * (cmax - c[i]);
    }
    // Camera -> output primaries, exposure.
    float rgb[3];
    for (int i = 0; i < 3; ++i)
        rgb[i] = p.gainLin * (p.m[3 * i] * c[0] + p.m[3 * i + 1] * c[1] + p.m[3 * i + 2] * c[2]);
    // Tone controls on luminance in log2 stops around 0.18 (hue preserving).
    float Y = p.lum[0] * rgb[0] + p.lum[1] * rgb[1] + p.lum[2] * rgb[2];
    if (p.toneActive && Y > 1e-6f) {
        float L = log2f(Y / 0.18f), L2 = L;
        L2 = L2 * p.contrast;
        L2 += p.midtones * expf(-sqr(L / 2.5f));
        L2 += p.highlights * smooth(0.f, 4.f, L) * 2.f;
        L2 += p.shadows * smooth(0.f, 5.f, -L) * 2.f;
        float ratio = exp2f(L2 - L);
        for (int i = 0; i < 3; ++i) rgb[i] *= ratio;
        Y *= ratio;
    }
    // Colour boost (vibrance) then saturation.
    if (p.colorBoost != 0.f || p.saturation != 1.f) {
        float mx = fmaxf(rgb[0], fmaxf(rgb[1], rgb[2])), mn = fminf(rgb[0], fminf(rgb[1], rgb[2]));
        float sat = mx > 1e-6f ? clampf((mx - mn) / mx, 0.f, 1.f) : 0.f;
        float f = p.saturation * (1.f + p.colorBoost * (1.f - sat));
        for (int i = 0; i < 3; ++i) rgb[i] = Y + (rgb[i] - Y) * f;
    }
    // Gamut mapping: compress negative/out-of-gamut components towards the achromatic axis.
    if (p.gamutMap) {
        float ach = fmaxf(rgb[0], fmaxf(rgb[1], rgb[2]));
        if (ach > 0.f)
            for (int i = 0; i < 3; ++i) {
                float d = (ach - rgb[i]) / ach;           // 0 = on axis, >1 = negative component
                const float th = 0.8f, lim = 1.2f;
                if (d > th) {
                    float x = (d - th) / (lim - th);
                    float cd = th + (lim - th) * x / (1.f + x);  // smooth approach to lim
                    rgb[i] = ach - cd * ach;
                }
            }
    }
    // Pre tone curve: filmic shoulder for display encodings (scene 1.0 -> 0.9).
    if (p.preToneCurve)
        for (int i = 0; i < 3; ++i) {
            float v = fmaxf(rgb[i], 0.f);
            rgb[i] = v < 0.6f ? v : 0.6f + 0.4f * (1.f - expf(-(v - 0.6f) / 0.4f));
        }
    // Output encoding, lift/gain in a gamma-2.4 domain for linear output.
    const bool adjust = p.lift != 0.f || p.gain != 1.f || p.softClip;
    for (int i = 0; i < 3; ++i) {
        float v = rgb[i];
        bool lin = p.gamma == 0;
        if (lin && !adjust) continue;
        float e = encode(v, lin ? 2 : p.gamma);
        if (p.lift != 0.f || p.gain != 1.f) e = e * p.gain + p.lift * (1.f - fminf(fmaxf(e, 0.f), 1.f));
        if (p.softClip && e > 0.8f) e = 0.8f + 0.2f * tanhf((e - 0.8f) / 0.2f);
        rgb[i] = lin ? decode_g24(e) : e;
    }
    o[0] = rgb[0]; o[1] = rgb[1]; o[2] = rgb[2]; o[3] = 1.f;
}
