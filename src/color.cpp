#include "color.h"

#include <algorithm>
#include <cmath>

namespace sfp {
namespace {

struct Ruvt { double r, u, v, t; };
// Robertson isotemperature lines, as used by the DNG SDK (dng_temperature.cpp).
const Ruvt kTemp[31] = {
    {0, 0.18006, 0.26352, -0.24341},   {10, 0.18066, 0.26589, -0.25479},  {20, 0.18133, 0.26846, -0.26876},
    {30, 0.18208, 0.27119, -0.28539},  {40, 0.18293, 0.27407, -0.30470},  {50, 0.18388, 0.27709, -0.32675},
    {60, 0.18494, 0.28021, -0.35156},  {70, 0.18611, 0.28342, -0.37915},  {80, 0.18740, 0.28668, -0.40955},
    {90, 0.18880, 0.28997, -0.44278},  {100, 0.19032, 0.29326, -0.47888}, {125, 0.19462, 0.30141, -0.58204},
    {150, 0.19962, 0.30921, -0.70471}, {175, 0.20525, 0.31647, -0.84901}, {200, 0.21142, 0.32312, -1.0182},
    {225, 0.21807, 0.32909, -1.2168},  {250, 0.22511, 0.33439, -1.4512},  {275, 0.23247, 0.33904, -1.7298},
    {300, 0.24010, 0.34308, -2.0637},  {325, 0.24792, 0.34655, -2.4681},  {350, 0.25591, 0.34951, -2.9641},
    {375, 0.26400, 0.35200, -3.5814},  {400, 0.27218, 0.35407, -4.3633},  {425, 0.28039, 0.35577, -5.3762},
    {450, 0.28863, 0.35714, -6.7262},  {475, 0.29685, 0.35823, -8.5955},  {500, 0.30505, 0.35907, -11.324},
    {525, 0.31320, 0.35968, -15.628},  {550, 0.32129, 0.36011, -23.325},  {575, 0.32931, 0.36038, -40.770},
    {600, 0.33724, 0.36051, -116.45}};
const double kTintScale = -3000.0;

Mat3 mul(const Mat3& a, const Mat3& b) {
    Mat3 r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) r.m[i * 3 + j] += a.m[i * 3 + k] * b.m[k * 3 + j];
    return r;
}
void mulv(const Mat3& a, const double v[3], double o[3]) {
    for (int i = 0; i < 3; ++i) o[i] = a.m[i * 3] * v[0] + a.m[i * 3 + 1] * v[1] + a.m[i * 3 + 2] * v[2];
}
Mat3 inv(const Mat3& a) {
    const double* m = a.m;
    double c0 = m[4] * m[8] - m[5] * m[7], c1 = m[5] * m[6] - m[3] * m[8], c2 = m[3] * m[7] - m[4] * m[6];
    double det = m[0] * c0 + m[1] * c1 + m[2] * c2;
    Mat3 r{};
    if (std::fabs(det) < 1e-12) return Mat3{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
    double d = 1.0 / det;
    r.m[0] = c0 * d; r.m[1] = (m[2] * m[7] - m[1] * m[8]) * d; r.m[2] = (m[1] * m[5] - m[2] * m[4]) * d;
    r.m[3] = c1 * d; r.m[4] = (m[0] * m[8] - m[2] * m[6]) * d; r.m[5] = (m[2] * m[3] - m[0] * m[5]) * d;
    r.m[6] = c2 * d; r.m[7] = (m[1] * m[6] - m[0] * m[7]) * d; r.m[8] = (m[0] * m[4] - m[1] * m[3]) * d;
    return r;
}
Mat3 diag(double a, double b, double c) { return Mat3{{a, 0, 0, 0, b, 0, 0, 0, c}}; }

void xy_to_XYZ(double x, double y, double o[3]) { o[0] = x / y; o[1] = 1; o[2] = (1 - x - y) / y; }
void XYZ_to_xy(const double v[3], double& x, double& y) {
    double s = v[0] + v[1] + v[2];
    if (s <= 0) { x = 0.3457; y = 0.3585; return; }
    x = v[0] / s; y = v[1] / s;
}

// Bradford chromatic adaptation from white w1 to w2 (xy).
Mat3 bradford(double x1, double y1, double x2, double y2) {
    const Mat3 B{{0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296}};
    double a[3], b[3], ca[3], cb[3];
    xy_to_XYZ(x1, y1, a); xy_to_XYZ(x2, y2, b);
    mulv(B, a, ca); mulv(B, b, cb);
    Mat3 s = diag(cb[0] / ca[0], cb[1] / ca[1], cb[2] / ca[2]);
    return mul(inv(B), mul(s, B));
}

// RGB -> XYZ for primaries and white (xy).
Mat3 rgb_to_xyz(double rx, double ry, double gx, double gy, double bx, double by, double wx, double wy) {
    Mat3 P{{rx / ry, gx / gy, bx / by, 1, 1, 1, (1 - rx - ry) / ry, (1 - gx - gy) / gy, (1 - bx - by) / by}};
    double w[3], s[3];
    xy_to_XYZ(wx, wy, w);
    mulv(inv(P), w, s);
    return mul(P, diag(s[0], s[1], s[2]));
}

struct Space { double p[6]; double wx, wy; };
Space space(Primaries p) {
    const double d65x = 0.3127, d65y = 0.3290, acesx = 0.32168, acesy = 0.33767;
    switch (p) {
        case Primaries::P3D65: return {{0.680, 0.320, 0.265, 0.690, 0.150, 0.060}, d65x, d65y};
        case Primaries::Rec2020: return {{0.708, 0.292, 0.170, 0.797, 0.131, 0.046}, d65x, d65y};
        case Primaries::DaVinciWideGamut: return {{0.8000, 0.3130, 0.1682, 0.9877, 0.0790, -0.1155}, d65x, d65y};
        case Primaries::ACES_AP0: return {{0.7347, 0.2653, 0.0000, 1.0000, 0.0001, -0.0770}, acesx, acesy};
        case Primaries::ACES_AP1: return {{0.713, 0.293, 0.165, 0.830, 0.128, 0.044}, acesx, acesy};
        default: return {{0.640, 0.330, 0.300, 0.600, 0.150, 0.060}, d65x, d65y};
    }
}

double illuminant_temp(int code) {
    switch (code) {
        case 1: case 4: case 9: return 5500;   // daylight, flash, fine weather
        case 2: case 14: return 4150;          // fluorescent, cool white
        case 3: return 2850;                   // tungsten
        case 10: return 6500;                  // cloudy
        case 11: return 7500;                  // shade
        case 12: return 6430;                  // daylight fluorescent
        case 13: return 5000;                  // day white fluorescent
        case 15: return 3450;                  // white fluorescent
        case 17: return 2856;                  // standard A
        case 18: return 4874;                  // standard B
        case 19: return 6774;                  // standard C
        case 20: return 5503;                  // D55
        case 21: return 6504;                  // D65
        case 22: return 7504;                  // D75
        case 23: return 5003;                  // D50
        case 24: return 3200;                  // ISO studio tungsten
        default: return 0;
    }
}

// XYZ -> camera for white xy (interpolate ColorMatrix1/2 in inverse temperature).
Mat3 color_matrix(const DngInfo& d, double x, double y) {
    Mat3 c1, c2;
    for (int i = 0; i < 9; ++i) { c1.m[i] = d.cm1[i]; c2.m[i] = d.hasCM2 ? d.cm2[i] : d.cm1[i]; }
    double t1 = illuminant_temp(d.illum1), t2 = illuminant_temp(d.illum2);
    Mat3 cm = c1;
    if (d.hasCM2 && t1 > 0 && t2 > 0 && t1 != t2) {
        if (t1 > t2) { std::swap(t1, t2); std::swap(c1, c2); }
        double t, tint;
        xy_to_temp_tint(x, y, t, tint);
        double g = t <= t1 ? 1.0 : t >= t2 ? 0.0 : (1.0 / t - 1.0 / t2) / (1.0 / t1 - 1.0 / t2);
        for (int i = 0; i < 9; ++i) cm.m[i] = g * c1.m[i] + (1 - g) * c2.m[i];
    }
    Mat3 ab = diag(d.analog[0], d.analog[1], d.analog[2]);
    return mul(ab, cm);
}

}  // namespace

void xy_to_temp_tint(double x, double y, double& temperature, double& tint) {
    double u = 2.0 * x / (1.5 - x + 6.0 * y);
    double v = 3.0 * y / (1.5 - x + 6.0 * y);
    double last_dt = 0, last_du = 0, last_dv = 0;
    temperature = 5000; tint = 0;
    for (int i = 1; i <= 30; ++i) {
        double du = 1.0, dv = kTemp[i].t;
        double len = std::sqrt(1.0 + dv * dv);
        du /= len; dv /= len;
        double uu = u - kTemp[i].u, vv = v - kTemp[i].v;
        double dt = -uu * dv + vv * du;
        if (dt <= 0 || i == 30) {
            if (dt > 0) dt = 0;
            dt = -dt;
            double f = i == 1 ? 0.0 : dt / (last_dt + dt);
            temperature = 1.0e6 / (kTemp[i - 1].r * f + kTemp[i].r * (1.0 - f));
            uu = u - (kTemp[i - 1].u * f + kTemp[i].u * (1.0 - f));
            vv = v - (kTemp[i - 1].v * f + kTemp[i].v * (1.0 - f));
            du = du * (1.0 - f) + last_du * f;
            dv = dv * (1.0 - f) + last_dv * f;
            len = std::sqrt(du * du + dv * dv);
            du /= len; dv /= len;
            tint = (uu * du + vv * dv) * kTintScale;
            break;
        }
        last_dt = dt; last_du = du; last_dv = dv;
    }
}

void temp_tint_to_xy(double temperature, double tint, double& x, double& y) {
    double r = 1.0e6 / temperature, offset = tint * (1.0 / kTintScale);
    for (int i = 0; i <= 29; ++i) {
        if (r < kTemp[i + 1].r || i == 29) {
            double f = (kTemp[i + 1].r - r) / (kTemp[i + 1].r - kTemp[i].r);
            double u = kTemp[i].u * f + kTemp[i + 1].u * (1.0 - f);
            double v = kTemp[i].v * f + kTemp[i + 1].v * (1.0 - f);
            double uu1 = 1.0, vv1 = kTemp[i].t, len1 = std::sqrt(1.0 + vv1 * vv1);
            uu1 /= len1; vv1 /= len1;
            double uu2 = 1.0, vv2 = kTemp[i + 1].t, len2 = std::sqrt(1.0 + vv2 * vv2);
            uu2 /= len2; vv2 /= len2;
            double uu3 = uu1 * f + uu2 * (1.0 - f), vv3 = vv1 * f + vv2 * (1.0 - f);
            double len3 = std::sqrt(uu3 * uu3 + vv3 * vv3);
            uu3 /= len3; vv3 /= len3;
            u += uu3 * offset; v += vv3 * offset;
            x = 1.5 * u / (u - 4.0 * v + 2.0);
            y = v / (u - 4.0 * v + 2.0);
            return;
        }
    }
}

ColorSetup color_setup(const DngInfo& d, bool asShot, double temp, double tint, Primaries out) {
    const double d50x = 0.3457, d50y = 0.3585;
    double wx = d50x, wy = d50y;
    double neutral[3];
    if (asShot) {
        for (int i = 0; i < 3; ++i) neutral[i] = std::max(1e-6, d.neutral[i]);
        // White xy from camera neutral: iterate matrix interpolation (DNG spec).
        for (int it = 0; it < 30; ++it) {
            double XYZ[3];
            mulv(inv(color_matrix(d, wx, wy)), neutral, XYZ);
            double nx, ny;
            XYZ_to_xy(XYZ, nx, ny);
            if (std::fabs(nx - wx) < 1e-7 && std::fabs(ny - wy) < 1e-7) { wx = nx; wy = ny; break; }
            wx = nx; wy = ny;
        }
    } else {
        temp_tint_to_xy(std::clamp(temp, 1500.0, 50000.0), std::clamp(tint, -150.0, 150.0), wx, wy);
        double XYZ[3];
        xy_to_XYZ(wx, wy, XYZ);
        mulv(color_matrix(d, wx, wy), XYZ, neutral);
        for (double& n : neutral) n = std::max(1e-6, n);
    }
    for (double& n : neutral) n = n / neutral[1] * 1.0;
    ColorSetup s{};
    for (int i = 0; i < 3; ++i) s.wb[i] = static_cast<float>(1.0 / (neutral[i] / neutral[1]));
    xy_to_temp_tint(wx, wy, s.temp, s.tint);
    // PCS (XYZ D50) -> camera, normalised so D50 white maps to max component 1.
    Mat3 pcsToCam = mul(color_matrix(d, wx, wy), bradford(d50x, d50y, wx, wy));
    double d50[3], c[3];
    xy_to_XYZ(d50x, d50y, d50);
    mulv(pcsToCam, d50, c);
    double scale = std::max({c[0], c[1], c[2]});
    for (double& v : pcsToCam.m) v /= scale;
    Mat3 camToPcs = inv(pcsToCam);
    // White-balanced camera RGB (neutral -> 1,1,1) -> PCS: scale columns by neutral.
    Mat3 wbToPcs = mul(camToPcs, diag(neutral[0], neutral[1], neutral[2]));
    // Normalise so that white-balanced (1,1,1) maps to Y = 1.
    double one[3] = {1, 1, 1}, pcs[3];
    mulv(wbToPcs, one, pcs);
    for (double& v : wbToPcs.m) v /= pcs[1];
    Space sp = space(out);
    Mat3 toOut = inv(rgb_to_xyz(sp.p[0], sp.p[1], sp.p[2], sp.p[3], sp.p[4], sp.p[5], sp.wx, sp.wy));
    Mat3 full = mul(toOut, mul(bradford(d50x, d50y, sp.wx, sp.wy), wbToPcs));
    if (out == Primaries::XYZ_D65) full = mul(bradford(d50x, d50y, 0.3127, 0.3290), wbToPcs);
    for (int i = 0; i < 9; ++i) s.matrix[i] = static_cast<float>(full.m[i]);
    Mat3 toXYZ = rgb_to_xyz(sp.p[0], sp.p[1], sp.p[2], sp.p[3], sp.p[4], sp.p[5], sp.wx, sp.wy);
    for (int i = 0; i < 3; ++i) s.lum[i] = static_cast<float>(out == Primaries::XYZ_D65 ? (i == 1) : toXYZ.m[3 + i]);
    return s;
}

}  // namespace sfp
