#include "gyro.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>

#include "dng.h"
#include "frame_cache.h"
#include "platform.h"

namespace sfp {

namespace {

const double kPi = 3.14159265358979323846;
// Sensor width the raster stands for, used for the pixel pitch when the file does not
// carry one. Calibrated, not nominal: on A001_092 (3264 raster, 28 mm) the picture moves by
// 49.1 px per degree of gyro rotation in x and y, which is 28 mm over a 32.5 mm wide raster.
// The fp's sensor is 35.9 mm wide (44.4 px per degree); the 10.5 % difference is not
// explained yet (lens, gyro scale or readout crop) and has been measured with one lens only.
const double kSensorWidthMm = 32.5;
// The other modes: on A001_001 (MQ, 3024 raster, 28 mm) the picture moves by 0.906 of what
// 32.5 mm predicts, which is the fp's full sensor width (MQ is the whole sensor binned by 2;
// HQ reads a narrower part of it).
const double kFullSensorWidthMm = 35.9;
const int kHqRasterWidth = 3264;
// Mark delay = readout/2 + exposure/2 + this. Fitted on the picture: +0.3 ms (A001_001, MQ 50,
// 1/640 s), -1.6 to +0.5 ms (A001_002, same mode, noisier), +0.8 ms (A001_092, HQ 25, 1/100 s).
// All within the scatter of the measurements, so no constant is added.
const double kMarkExtraS = 0.0;
const double kMaxAutoZoom = 4.0;

Quat mul(const Quat& a, const Quat& b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
Quat conj(const Quat& a) { return {a.w, -a.x, -a.y, -a.z}; }
double dot(const Quat& a, const Quat& b) { return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z; }
Quat normalized(const Quat& a) {
    double n = std::sqrt(dot(a, a));
    if (n < 1e-300) return {};
    return {a.w / n, a.x / n, a.y / n, a.z / n};
}
// Rotation by the vector v (radians, axis * angle).
Quat from_vector(double x, double y, double z) {
    double a = std::sqrt(x * x + y * y + z * z);
    if (a < 1e-12) return normalized({1, x / 2, y / 2, z / 2});
    double s = std::sin(a / 2) / a;
    return {std::cos(a / 2), x * s, y * s, z * s};
}
void to_matrix(const Quat& q, float* m) {
    double w = q.w, x = q.x, y = q.y, z = q.z;
    m[0] = static_cast<float>(1 - 2 * (y * y + z * z)); m[1] = static_cast<float>(2 * (x * y - w * z)); m[2] = static_cast<float>(2 * (x * z + w * y));
    m[3] = static_cast<float>(2 * (x * y + w * z)); m[4] = static_cast<float>(1 - 2 * (x * x + z * z)); m[5] = static_cast<float>(2 * (y * z - w * x));
    m[6] = static_cast<float>(2 * (x * z - w * y)); m[7] = static_cast<float>(2 * (y * z + w * x)); m[8] = static_cast<float>(1 - 2 * (x * x + y * y));
}

uint16_t u16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t u32(const uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (static_cast<uint32_t>(p[3]) << 24); }

std::string upper(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}
std::string file_name(const std::string& p) {
    size_t s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}
std::string stem(const std::string& name) {
    size_t d = name.find_last_of('.');
    return d == std::string::npos ? name : name.substr(0, d);
}

}  // namespace

// ---- frame times ----

// Frame marks on a regular cadence. m[i] is the recorded mark of frame i (seconds), usable
// where ok[i]. The sensor's frames are equally spaced; a mark is the frame's cadence time
// plus how late the camera's recorder task was, which is small and steady for most frames
// and tens of milliseconds for a few. So the cadence is the LOWER envelope of the marks:
//   - period T: the nominal frame period, corrected by the slope of the envelope;
//   - a dropped frame (the sensor's cadence went on, no frame was recorded) is a lasting step
//     of a whole period in the marks, unlike a late mark, which is made up within a few
//     frames; steps are kept as gaps in the cadence;
//   - slow drift between the frame clock and the sample clock: the envelope is a rolling
//     minimum over +-2 s, averaged over the same window (still a lower bound);
//   - the result is raised by the median lateness of the clip's marks, so that it stands
//     for an undisturbed mark, which is what the mark delay is measured against.
static void regular_marks(const std::vector<double>& m, const std::vector<uint8_t>& ok, double fps, std::vector<double>& out,
                          int& drops, double& typicalLate, int& late) {
    const int n = static_cast<int>(m.size());
    out = m;
    drops = late = 0;
    typicalLate = 0;
    std::vector<int> idx;
    for (int i = 0; i < n; ++i)
        if (ok[i]) idx.push_back(i);
    if (idx.size() < 8) return;
    std::vector<double> steps;
    for (size_t a = 1; a < idx.size(); ++a) steps.push_back((m[idx[a]] - m[idx[a - 1]]) / (idx[a] - idx[a - 1]));
    std::nth_element(steps.begin(), steps.begin() + steps.size() / 2, steps.end());
    const double median = steps[steps.size() / 2];
    if (!(median > 1e-4)) return;
    double T = fps > 1 && std::fabs(median * fps - 1) < 0.02 ? 1.0 / fps : median;
    const int W = std::max(5, static_cast<int>(std::lround(2.0 / T)));       // envelope window, frames each side
    const int Wd = std::max(5, static_cast<int>(std::lround(0.5 / T)));      // a late mark is made up well within this
    std::vector<double> r(n, 0.0), lo(n), env(n);
    std::vector<int> cadence(n);
    for (int pass = 0; pass < 3; ++pass) {
        // Dropped frames: the earliest marks ahead sit a whole number of periods above the earliest behind.
        int cum = 0;
        for (int i = 0; i < n; ++i) {
            if (ok[i] && i > 0) {
                double behind = 1e300, ahead = 1e300;
                int na = 0;
                for (int j = i - 1; j >= std::max(0, i - Wd); --j)
                    if (ok[j]) behind = std::min(behind, r[j]);
                for (int j = i; j <= std::min(n - 1, i + Wd); ++j)
                    if (ok[j]) { ahead = std::min(ahead, m[j] - (j + cum) * T); ++na; }
                if (behind < 1e299 && na >= 5) {
                    const int jump = static_cast<int>(std::floor((ahead - behind) / T + 0.5));
                    if (jump >= 1) cum += jump;
                }
            }
            cadence[i] = i + cum;
            r[i] = m[i] - cadence[i] * T;
        }
        drops = cum;
        for (int i = 0; i < n; ++i) {
            double v = 1e300;
            for (int j = std::max(0, i - W); j <= std::min(n - 1, i + W); ++j)
                if (ok[j]) v = std::min(v, r[j]);
            lo[i] = v;
        }
        // Where a whole window has no mark: the nearest value.
        for (int i = 1; i < n; ++i)
            if (lo[i] > 1e299) lo[i] = lo[i - 1];
        for (int i = n - 2; i >= 0; --i)
            if (lo[i] > 1e299) lo[i] = lo[i + 1];
        for (int i = 0; i < n; ++i) {
            double sum = 0;
            int c = 0;
            for (int j = std::max(0, i - W); j <= std::min(n - 1, i + W); ++j) { sum += lo[j]; ++c; }
            env[i] = sum / c;
        }
        // The envelope's remaining slope is the error of the period.
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < n; ++i) { sx += cadence[i]; sy += env[i]; sxx += static_cast<double>(cadence[i]) * cadence[i]; sxy += cadence[i] * env[i]; }
        const double den = n * sxx - sx * sx;
        const double slope = den > 0 ? (n * sxy - sx * sy) / den : 0;
        if (std::fabs(slope) < 1e-9 || pass == 2) break;
        T += slope;
    }
    std::vector<double> lateness;
    for (int i : idx) lateness.push_back(r[i] - env[i]);
    std::vector<double> sorted = lateness;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    typicalLate = sorted[sorted.size() / 2];
    for (double v : lateness)
        if (v - typicalLate > 0.005) ++late;
    for (int i = 0; i < n; ++i) out[i] = cadence[i] * T + env[i] + typicalLate;
}

// ---- file ----

std::shared_ptr<GyroClip> GyroClip::parse(const uint8_t* d, size_t size, std::string& e) {
    if (size < 128 || std::memcmp(d, "FPGY", 4) != 0) { e = "not an FPGY file"; return nullptr; }
    auto c = std::make_shared<GyroClip>();
    GyroHeader& h = c->h;
    h.version = u16(d + 4);
    const size_t header = u16(d + 6);
    if (h.version != 1) { e = "unsupported FPGY version " + std::to_string(h.version); return nullptr; }
    if (header < 128 || header > size) { e = "bad FPGY header size"; return nullptr; }
    h.frameCount = u32(d + 8);
    h.sampleCount = u32(d + 12);
    h.rateHz = u32(d + 16) / 1000.0;
    float lsb;
    std::memcpy(&lsb, d + 20, 4);
    h.lsbPerDps = lsb;
    h.rasterW = u16(d + 24); h.rasterH = u16(d + 26);
    h.activeX = u16(d + 28); h.activeY = u16(d + 30); h.activeW = u16(d + 32); h.activeH = u16(d + 34);
    const uint32_t num = u32(d + 36), den = u32(d + 40);
    h.fps = den ? static_cast<double>(num) / den : 0;
    h.readoutS = u32(d + 44) * 1e-6;
    h.exposureS = u32(d + 48) * 1e-6;
    h.focalMm = u32(d + 52) * 1e-3;
    h.markDelayS = static_cast<int32_t>(u32(d + 56)) * 1e-6;
    for (int i = 0; i < 3; ++i) h.axis[i] = static_cast<int8_t>(d[60 + i]);
    h.flags = d[63];
    h.preroll = u32(d + 64);
    h.pitchNm = u32(d + 68);
    h.resolution = u32(d + 72); h.dcCrop = u32(d + 76); h.bitDepth = u32(d + 80); h.sensorMode = u32(d + 84);

    const uint64_t need = header + 4ull * h.frameCount + 6ull * h.sampleCount;
    if (need > size) { e = "FPGY file is truncated"; return nullptr; }
    if (h.frameCount < 1 || h.sampleCount < 2) { e = "FPGY file has no frames or samples"; return nullptr; }
    if (!(h.rateHz > 1) || !std::isfinite(h.lsbPerDps) || !(h.lsbPerDps > 0)) { e = "bad FPGY sample rate or scale"; return nullptr; }
    if (h.rasterW < 16 || h.rasterH < 16) { e = "bad FPGY raster size"; return nullptr; }
    bool used[3] = {false, false, false};
    for (int i = 0; i < 3; ++i) {
        int a = std::abs(h.axis[i]);
        if (a < 1 || a > 3 || used[a - 1]) { e = "bad FPGY axis map"; return nullptr; }
        used[a - 1] = true;
    }
    if (h.activeW <= 0 || h.activeH <= 0 || h.activeX + h.activeW > h.rasterW || h.activeY + h.activeH > h.rasterH) {
        h.activeX = h.activeY = 0; h.activeW = h.rasterW; h.activeH = h.rasterH;
    }
    std::vector<uint32_t> table(h.frameCount);
    for (uint32_t i = 0; i < h.frameCount; ++i) table[i] = u32(d + header + 4ull * i);
    std::vector<int16_t> samples(3ull * h.sampleCount);
    const uint8_t* sp = d + header + 4ull * h.frameCount;
    for (size_t i = 0; i < samples.size(); ++i) samples[i] = static_cast<int16_t>(u16(sp + 2 * i));
    return assemble(h, std::move(table), std::vector<uint8_t>(h.frameCount, 1), samples.data(), e);
}

std::shared_ptr<GyroClip> GyroClip::assemble(const GyroHeader& hdr, std::vector<uint32_t> table, std::vector<uint8_t> valid,
                                             const int16_t* samples, std::string& e) {
    auto c = std::make_shared<GyroClip>();
    c->h = hdr;
    GyroHeader& h = c->h;
    if (table.size() != h.frameCount || valid.size() != h.frameCount || h.frameCount < 1 || h.sampleCount < 2) { e = "bad gyro track"; return nullptr; }
    uint32_t prev = 0;
    for (uint32_t v : table) {
        if (v < prev || v > h.sampleCount) { e = "bad FPGY frame table"; return nullptr; }
        prev = v;
    }
    c->table_ = std::move(table);
    c->valid_ = std::move(valid);
    {
        std::vector<double> marks(c->table_.size());
        for (size_t i = 0; i < marks.size(); ++i) marks[i] = c->table_[i] / h.rateHz;
        double typical = 0;
        regular_marks(marks, c->valid_, h.fps, c->mark_, c->droppedFrames, typical, c->lateMarks);
        c->typicalLateMs = typical * 1e3;
    }
    // Integrate the angular rate (trapezoid between samples) into camera -> world
    // orientations. Camera axes: x right, y down, z into the scene. The file's pan and
    // tilt are the motion of the camera's aim along image x (right) and image y (down):
    // aiming right is a turn about +y, aiming down a turn about -x; roll is the turn about
    // +z (clockwise seen from behind the camera). Checked on A001_092: the picture moves
    // by -49 px per degree of pan, and turns anticlockwise for positive roll.
    const double k = kPi / 180.0 / h.lsbPerDps, dt = 1.0 / h.rateHz;
    auto rate = [&](uint32_t i, double w[3]) {
        double g[3];
        for (int a = 0; a < 3; ++a) g[a] = samples[3ull * i + a] * k;
        double v[3];
        for (int a = 0; a < 3; ++a) v[a] = (h.axis[a] < 0 ? -1.0 : 1.0) * g[std::abs(h.axis[a]) - 1];
        w[0] = -v[1]; w[1] = v[0]; w[2] = v[2];
    };
    c->q_.resize(h.sampleCount);
    double a[3], b[3];
    rate(0, a);
    Quat q;
    c->q_[0] = q;
    for (uint32_t i = 1; i < h.sampleCount; ++i) {
        rate(i, b);
        q = normalized(mul(q, from_vector((a[0] + b[0]) * 0.5 * dt, (a[1] + b[1]) * 0.5 * dt, (a[2] + b[2]) * 0.5 * dt)));
        c->q_[i] = q;
        std::memcpy(a, b, sizeof a);
    }
    for (uint8_t v : c->valid_) c->noData += v ? 0 : 1;
    return c;
}

double GyroClip::exposure_s(long long i) const {
    if (exposureByFrame.empty()) return h.exposureS;
    return exposureByFrame[static_cast<size_t>(std::clamp<long long>(i, 0, static_cast<long long>(exposureByFrame.size()) - 1))];
}

double GyroClip::delay_s(long long i) const {
    const double e = exposure_s(i);
    if (e > 0 && h.readoutS > 0) return h.readoutS / 2 + e / 2 + kMarkExtraS;
    return h.markDelayS;
}

double GyroClip::frame_time(long long i, double syncMs, bool rawMarks) const {
    i = std::clamp<long long>(i, 0, frames() - 1);
    const double mark = rawMarks ? table_[static_cast<size_t>(i)] / h.rateHz : mark_[static_cast<size_t>(i)];
    return mark - (delay_s(i) + syncMs * 1e-3);
}

Quat GyroClip::orientation(double t) const {
    double s = std::clamp(t * h.rateHz, 0.0, static_cast<double>(q_.size() - 1));
    size_t i = std::min(static_cast<size_t>(s), q_.size() - 2);
    double f = s - static_cast<double>(i);
    const Quat& a = q_[i];
    Quat b = q_[i + 1];
    if (dot(a, b) < 0) b = {-b.w, -b.x, -b.y, -b.z};
    return normalized({a.w + f * (b.w - a.w), a.x + f * (b.x - a.x), a.y + f * (b.y - a.y), a.z + f * (b.z - a.z)});
}

// ---- path smoothing ----

void GyroClip::used_range(const StabSettings& s, int& first, int& last) const {
    const int n = frames();
    first = 0;
    last = n - 1;
    if (s.rangeLast < 0) return;
    const long long a = std::clamp<long long>(s.rangeFirst, 0, n - 1), b = std::clamp<long long>(s.rangeLast, 0, n - 1);
    if (a > b) return;
    first = static_cast<int>(a);
    last = static_cast<int>(b);
}

void GyroClip::ensure_path(const StabSettings& s) const {
    const double sigma = std::max(0.0, s.smoothness);
    int first, last;
    used_range(s, first, last);
    if (path_.valid && path_.smoothness == sigma && path_.syncMs == s.syncMs && path_.first == first && path_.last == last &&
        path_.rawMarks == s.rawMarks) return;
    const int n = frames();
    path_.time.resize(n);
    path_.actual.resize(n);
    for (int i = 0; i < n; ++i) {
        path_.time[i] = frame_time(i, s.syncMs, s.rawMarks);
        path_.actual[i] = orientation(path_.time[i]);
    }
    path_.smoothness = sigma;
    path_.syncMs = s.syncMs;
    path_.rawMarks = s.rawMarks;
    path_.first = first;
    path_.last = last;
    smooth_path(false, false, path_.smooth);
    path_.valid = true;
    zoom_.valid = false;
}

// Zero-phase Gaussian low-pass of the orientation (normalised weighted quaternion mean over
// +-3 sigma, truncated and renormalised at the ends). The frames before the used range, the
// range and the frames after it are three pieces. Each end of the range is either a clip end
// (the mean stops there) or the smoothing runs on into the neighbouring piece.
void GyroClip::smooth_path(bool pastStart, bool pastEnd, std::vector<Quat>& out) const {
    const int n = frames();
    const double sigma = path_.smoothness;
    const int first = path_.first, last = path_.last;
    const std::vector<double>& t = path_.time;
    out = path_.actual;
    if (!(sigma > 1e-6)) return;
    for (int i = 0; i < n; ++i) {
        int lo = i < first ? 0 : (i > last ? last + 1 : first), hi = i < first ? first - 1 : (i > last ? n - 1 : last);
        if (i >= first && i <= last) {
            if (pastStart) lo = 0;
            if (pastEnd) hi = n - 1;
        }
        const Quat& ref = path_.actual[i];
        Quat acc{0, 0, 0, 0};
        auto add = [&](int j) {
            double u = (t[j] - t[i]) / sigma;
            if (std::fabs(u) > 3.0) return false;
            double w = std::exp(-0.5 * u * u);
            const Quat& q = path_.actual[j];
            if (dot(ref, q) < 0) w = -w;
            acc.w += w * q.w; acc.x += w * q.x; acc.y += w * q.y; acc.z += w * q.z;
            return true;
        };
        for (int j = i; j <= hi && add(j); ++j) {}
        for (int j = i - 1; j >= lo && add(j); --j) {}
        out[i] = normalized(acc);
    }
}

void GyroClip::range_ends(const StabSettings& s, bool& pastStart, bool& pastEnd) const {
    pastStart = pastEnd = false;
    const double f = focal_px(s);
    if (!(f > 0)) return;
    std::lock_guard<std::mutex> l(m_);
    ensure_zoom(s, f);
    pastStart = zoom_.pastStart;
    pastEnd = zoom_.pastEnd;
}

Quat GyroClip::smoothed(long long i, const StabSettings& s) const {
    std::lock_guard<std::mutex> l(m_);
    ensure_zoom(s, focal_px(s));
    return zoom_.virt[static_cast<size_t>(std::clamp<long long>(i, 0, frames() - 1))];
}

int GyroClip::limited_frames(const StabSettings& s) const {
    std::lock_guard<std::mutex> l(m_);
    ensure_zoom(s, focal_px(s));
    return zoom_.limited;
}

void GyroClip::rotations(long long i, const StabSettings& s, double rs, const Quat& virt, float* rot) const {
    const double t = frame_time(i, s.syncMs, s.rawMarks);
    const double span = rs * readout_s(s);
    for (int k = 0; k < STAB_KNOTS; ++k) {
        // Real camera at the time this row was read <- virtual camera.
        double tk = t + span * (static_cast<double>(k) / (STAB_KNOTS - 1) - 0.5);
        to_matrix(normalized(mul(conj(orientation(tk)), virt)), rot + 9 * k);
    }
}

bool GyroClip::ends_early(const StabSettings& s) const {
    return s.clipFrames > 0 ? s.clipFrames > frames() : (h.flags & 1) != 0;
}

double GyroClip::focal_px(const StabSettings& s, std::string* why) const {
    const double mm = s.focalMm > 0 ? s.focalMm : h.focalMm;
    if (!(mm > 0)) {
        if (why) *why = "the file has no focal length (manual lens): set Focal Length";
        return 0;
    }
    const double widthMm = h.rasterW == kHqRasterWidth ? kSensorWidthMm : kFullSensorWidthMm;
    double pitchMm = h.pitchNm > 0 ? h.pitchNm * 1e-6 : widthMm / (h.dcCrop ? 1.5 : 1.0) / h.rasterW;
    return mm / pitchMm;
}

// The virtual camera of every frame and the automatic zoom, over the used range.
// Fixed zoom: the smallest one that hides the borders on every frame of the range, up to the
// limit. Dynamic zoom: each frame's own need, widened to the largest need within the look-ahead
// time and eased over the same time, so it changes slowly and is never below the need.
// Frames that would need more than the limit get less correction instead: the virtual camera
// is moved towards the real one until the frame fits (eased over neighbouring frames), so a
// violent move costs stabilisation there instead of zoom everywhere.
// Next to frames without gyro data everything fades to none over one second.
void GyroClip::ensure_zoom(const StabSettings& s, double f) const {
    ensure_path(s);
    // The amount is 0..1 in the plug-in; negative values (reversed readout) are for measurements.
    const double rs = std::clamp(s.rollingShutter, -1.0, 1.0);
    const double limit = std::clamp(s.maxZoom, 1.0, kMaxAutoZoom);
    const bool fade = ends_early(s);
    const int mode = s.zoomMode == 1 ? 1 : 0;
    const double zsmooth = std::clamp(s.zoomSmooth, 0.1, 60.0);
    if (zoom_.valid && zoom_.autoZoom == s.autoZoom && zoom_.rs == rs && zoom_.focal == f && zoom_.maxZoom == limit && zoom_.fade == fade &&
        zoom_.mode == mode && (mode == 0 || zoom_.smooth == zsmooth) && zoom_.readout == readout_s(s)) return;
    const int n = frames();
    const int first = path_.first, last = path_.last;
    zoom_.virt = path_.smooth;
    zoom_.value = zoom_.low = zoom_.mean = 1;
    zoom_.limited = 0;
    zoom_.autoZoom = s.autoZoom; zoom_.rs = rs; zoom_.focal = f; zoom_.maxZoom = limit; zoom_.fade = fade;
    zoom_.mode = mode; zoom_.smooth = zsmooth; zoom_.readout = readout_s(s);
    zoom_.valid = true;
    StabParams p{};
    p.on = 1;
    p.focal = static_cast<float>(f);
    p.cx = h.rasterW * 0.5f; p.cy = h.rasterH * 0.5f;
    p.rows = static_cast<float>(h.rasterH);
    // Border points of the picture (active area).
    const int per = 8;
    std::vector<float> bx, by;
    const float x0 = static_cast<float>(h.activeX), y0 = static_cast<float>(h.activeY);
    const float x1 = x0 + h.activeW, y1 = y0 + h.activeH;
    for (int k = 0; k < per; ++k) {
        float u = static_cast<float>(k) / per;
        bx.push_back(x0 + u * (x1 - x0)); by.push_back(y0);
        bx.push_back(x1); by.push_back(y0 + u * (y1 - y0));
        bx.push_back(x1 - u * (x1 - x0)); by.push_back(y1);
        bx.push_back(x0); by.push_back(y1 - u * (y1 - y0));
    }
    auto inside = [&](double zoom) {
        p.invZoom = static_cast<float>(1.0 / zoom);
        for (size_t k = 0; k < bx.size(); ++k) {
            float sx, sy;
            stab_map(p, bx[k], by[k], &sx, &sy);
            if (sx < x0 || sy < y0 || sx > x1 || sy > y1) return false;
        }
        return true;
    };
    // The real camera at frame i blended towards q by the share a.
    auto blend = [&](int i, Quat q, double a) {
        const Quat& q0 = path_.actual[i];
        if (dot(q0, q) < 0) q = {-q.w, -q.x, -q.y, -q.z};
        return normalized({q0.w + a * (q.w - q0.w), q0.x + a * (q.x - q0.x), q0.y + a * (q.y - q0.y), q0.z + a * (q.z - q0.z)});
    };
    // Frames of one piece (before the range, the range, after it) do not affect each other.
    auto piece = [&](int i, int& lo, int& hi) {
        lo = i < first ? 0 : (i > last ? last + 1 : first);
        hi = i < first ? first - 1 : (i > last ? n - 1 : last);
    };
    const double fps = h.fps > 1 ? h.fps : 25.0;
    // The ends of a cut range. Stopping the smoothing at the cut makes the path lag where the
    // camera was moving through the cut; running it on lets a rough stretch just beyond the
    // cut pull on the path. Which is better depends on the footage, so each end is tried both
    // ways and the combination that needs the least zoom (first: the fewest frames over the
    // limit) is used.
    zoom_.pastStart = zoom_.pastEnd = false;
    std::vector<Quat> smoothPath = path_.smooth;
    if (f > 0 && (first > 0 || last < n - 1)) {
        long long best = -1;
        std::vector<Quat> trial;
        for (int v = 0; v < 4; ++v) {
            const bool ps = (v & 1) != 0, pe = (v & 2) != 0;
            if ((ps && first == 0) || (pe && last == n - 1)) continue;
            if (v == 0) trial = path_.smooth;
            else smooth_path(ps, pe, trial);
            int over = 0;
            double zoom = 1;
            for (int i = first; i <= last; ++i) {
                if (!valid_[i]) continue;
                rotations(i, s, rs, trial[i], p.rot);
                if (inside(zoom)) continue;
                if (!inside(limit)) { ++over; continue; }
                double lo = zoom, hi = limit;
                for (int it = 0; it < 10; ++it) {
                    double mid = 0.5 * (lo + hi);
                    (inside(mid) ? hi : lo) = mid;
                }
                zoom = hi;
            }
            const long long cost = over * 1000000LL + std::llround((zoom - 1) * 10000);
            if (best < 0 || cost < best) {
                best = cost;
                smoothPath.swap(trial);
                zoom_.pastStart = ps;
                zoom_.pastEnd = pe;
            }
        }
    }
    zoom_.virt = smoothPath;
    std::vector<double> zf(n, 1.0);                 // automatic zoom per frame
    std::vector<double> rsf(n, rs);                 // rolling-shutter amount per frame
    if (s.autoZoom && f > 0) {
        std::vector<uint8_t> reduce(n, 0);          // frames that do not fit at their zoom
        if (mode == 0) {
            double zoom = 1;
            bool over = false;
            for (int i = first; i <= last; ++i) {
                if (!valid_[i]) continue;
                rotations(i, s, rs, smoothPath[i], p.rot);
                if (inside(zoom)) continue;
                if (!inside(limit)) { reduce[i] = 1; over = true; continue; }
                double lo = zoom, hi = limit;
                for (int it = 0; it < 14; ++it) {
                    double mid = 0.5 * (lo + hi);
                    (inside(mid) ? hi : lo) = mid;
                }
                zoom = hi;
            }
            if (over) zoom = limit;
            std::fill(zf.begin(), zf.end(), zoom);
        } else {
            std::vector<double> need(n, 1.0);
            for (int i = first; i <= last; ++i) {
                if (!valid_[i]) continue;
                rotations(i, s, rs, smoothPath[i], p.rot);
                if (inside(1.0)) continue;
                if (!inside(limit)) { need[i] = limit; reduce[i] = 1; continue; }
                double lo = 1, hi = limit;
                for (int it = 0; it < 12; ++it) {
                    double mid = 0.5 * (lo + hi);
                    (inside(mid) ? hi : lo) = mid;
                }
                need[i] = hi;
            }
            // Largest need within +-w frames, then a Gaussian mean of that over the same +-w:
            // every value in a frame's window is at least the frame's own need, so is the mean.
            const int w = std::max(1, static_cast<int>(std::lround(zsmooth * fps)));
            std::vector<double> top(n, 1.0);
            for (int i = first; i <= last; ++i) {
                double v = 1;
                for (int j = std::max(first, i - w); j <= std::min(last, i + w); ++j) v = std::max(v, need[j]);
                top[i] = v;
            }
            for (int i = first; i <= last; ++i) {
                double sum = 0, wsum = 0;
                for (int j = std::max(first, i - w); j <= std::min(last, i + w); ++j) {
                    const double u = (j - i) / (0.5 * w), g = std::exp(-0.5 * u * u);
                    sum += g * top[j];
                    wsum += g;
                }
                zf[i] = sum / wsum;
            }
            for (int i = 0; i < first; ++i) zf[i] = zf[first];
            for (int i = last + 1; i < n; ++i) zf[i] = zf[last];
        }
        double sum = 0;
        int cnt = 0;
        zoom_.low = kMaxAutoZoom;
        for (int i = first; i <= last; ++i) {
            zoom_.value = std::max(zoom_.value, zf[i]);
            zoom_.low = std::min(zoom_.low, zf[i]);
            sum += zf[i];
            ++cnt;
        }
        zoom_.mean = cnt ? sum / cnt : 1;
        // Frames outside the range keep its zoom; where that is not enough they get less correction.
        for (int i = 0; i < n; ++i) {
            if ((i >= first && i <= last) || !valid_[i]) continue;
            rotations(i, s, rs, smoothPath[i], p.rot);
            if (!inside(zf[i])) reduce[i] = 1;
        }
        // Share of the correction that fits, per frame.
        std::vector<double> share(n, 1.0);
        bool any = false;
        for (int i = 0; i < n; ++i) {
            if (!reduce[i]) continue;
            double lo = 0, hi = 1;
            for (int it = 0; it < 10; ++it) {
                double mid = 0.5 * (lo + hi);
                rotations(i, s, rs, blend(i, smoothPath[i], mid), p.rot);
                (inside(zf[i]) ? lo : hi) = mid;
            }
            share[i] = lo;
            any = true;
            if (lo <= 0) {
                // Not even the rolling-shutter correction alone fits: take less of that as well.
                double a = 0, b = 1;
                for (int it = 0; it < 10; ++it) {
                    double mid = 0.5 * (a + b);
                    rotations(i, s, rs * mid, path_.actual[i], p.rot);
                    (inside(zf[i]) ? a : b) = mid;
                }
                rsf[i] = rs * a;
            }
        }
        if (any) {
            // Ease in and out: minimum over +-k frames, then the mean over +-k of that (never
            // above the frame's own share, so every frame still fits).
            const int k = std::max(1, static_cast<int>(std::lround(0.4 * fps)));
            std::vector<double> low(n), lowRs(n, 1.0);
            for (int i = 0; i < n; ++i) {
                int lo, hi;
                piece(i, lo, hi);
                double v = 1, r = 1;
                for (int j = std::max(lo, i - k); j <= std::min(hi, i + k); ++j) {
                    v = std::min(v, share[j]);
                    if (rs != 0) r = std::min(r, rsf[j] / rs);
                }
                low[i] = v;
                lowRs[i] = r;
            }
            for (int i = 0; i < n; ++i) {
                int lo, hi;
                piece(i, lo, hi);
                double sum2 = 0, sumRs = 0;
                int c2 = 0;
                for (int j = std::max(lo, i - k); j <= std::min(hi, i + k); ++j) { sum2 += low[j]; sumRs += lowRs[j]; ++c2; }
                rsf[i] = rs * sumRs / c2;
                if (sum2 / c2 < 1.0) {
                    zoom_.virt[i] = blend(i, smoothPath[i], sum2 / c2);
                    if (i >= first && i <= last) ++zoom_.limited;
                }
            }
        }
    }
    zoom_.zoom.assign(n, 1.0f);
    for (int i = 0; i < n; ++i) zoom_.zoom[i] = static_cast<float>(zf[i]);

    zoom_.rs_.assign(n, static_cast<float>(rs));
    for (int i = 0; i < n; ++i) zoom_.rs_[i] = static_cast<float>(rsf[i]);
    if ((fade || noData > 0) && n > 1) {
        // Fade to none over the second next to frames without gyro data (the end of a gyro
        // file that stops before the clip, or a stretch of frames without blocks): the frame
        // next to the gap is shown as recorded, like the frames in it.
        const int len = std::min(n - 1, std::max(1, static_cast<int>(std::lround(h.fps > 1 ? h.fps : 25.0))));
        std::vector<int> dist(n, n + len);
        int d = fade ? 0 : n + len;                       // frame n has no data when the file ends early
        for (int i = n - 1; i >= 0; --i) { d = valid_[i] ? std::min(d + 1, n + len) : 0; dist[i] = d; }
        d = n + len;
        for (int i = 0; i < n; ++i) { d = valid_[i] ? std::min(d + 1, n + len) : 0; dist[i] = std::min(dist[i], d); }
        for (int i = 0; i < n; ++i) {
            if (dist[i] > len) continue;
            double u = dist[i] > 0 ? static_cast<double>(dist[i] - 1) / len : 0.0;
            double g = u * u * (3 - 2 * u);
            const double z = 1 + (zf[i] - 1) * g;
            const Quat target = zoom_.virt[i];
            if (s.autoZoom && f > 0 && g > 0) {
                // The reduced zoom must still hide the border: take less correction if needed.
                rotations(i, s, rsf[i] * g, blend(i, target, g), p.rot);
                if (!inside(z)) {
                    double lo = 0, hi = g;
                    for (int it = 0; it < 10; ++it) {
                        double mid = 0.5 * (lo + hi);
                        rotations(i, s, rsf[i] * mid, blend(i, target, mid), p.rot);
                        (inside(z) ? lo : hi) = mid;
                    }
                    g = lo;
                }
            }
            zoom_.virt[i] = blend(i, target, g);
            zoom_.rs_[i] = static_cast<float>(rsf[i] * g);
            zoom_.zoom[i] = static_cast<float>(z);
        }
    }
}

double GyroClip::frame_zoom(long long i, const StabSettings& s) const {
    const double f = focal_px(s);
    if (!(f > 0) || i < 0 || i >= frames()) return 1;
    std::lock_guard<std::mutex> l(m_);
    ensure_zoom(s, f);
    return zoom_.zoom[static_cast<size_t>(i)];
}

void GyroClip::zoom_stats(const StabSettings& s, double& lo, double& mean, double& hi) const {
    lo = mean = hi = 1;
    const double f = focal_px(s);
    if (!(f > 0)) return;
    std::lock_guard<std::mutex> l(m_);
    ensure_zoom(s, f);
    lo = zoom_.low; mean = zoom_.mean; hi = zoom_.value;
}

double GyroClip::auto_zoom(const StabSettings& s) const {
    const double f = focal_px(s);
    if (!(f > 0)) return 1;
    std::lock_guard<std::mutex> l(m_);
    ensure_zoom(s, f);
    return zoom_.value;
}

bool GyroClip::warp(long long i, const StabSettings& s, StabParams& out, std::string& why) const {
    out.on = 0;
    if (!s.enable) { why = "switched off"; return false; }
    const double f = focal_px(s, &why);
    if (!(f > 0)) return false;
    out.focal = static_cast<float>(f);
    out.cx = h.rasterW * 0.5f; out.cy = h.rasterH * 0.5f;
    out.rows = static_cast<float>(h.rasterH);
    const double manual = std::clamp(s.zoom, 0.1, 10.0);
    if (!has_data(i)) {
        // No gyro data for this frame: the picture as recorded, with the manual zoom only.
        if (manual == 1.0) return true;
        out.invZoom = static_cast<float>(1.0 / manual);
        for (int k = 0; k < STAB_KNOTS; ++k) to_matrix(Quat{}, out.rot + 9 * k);
        out.on = 1;
        return true;
    }
    {
        std::lock_guard<std::mutex> l(m_);
        ensure_zoom(s, f);
        const size_t k = static_cast<size_t>(i);
        out.invZoom = static_cast<float>(1.0 / (zoom_.zoom[k] * manual));
        rotations(i, s, zoom_.rs_[k], zoom_.virt[k], out.rot);
    }
    out.on = 1;
    return true;
}

std::string GyroClip::status(const StabSettings& s) const {
    char buf[512];
    int n = inFrame ? std::snprintf(buf, sizeof buf, "gyro: %d frames, in-frame, %.2f Hz", frames(), h.rateHz)
                    : std::snprintf(buf, sizeof buf, "%s: %d frames, %.2f Hz", name.c_str(), frames(), h.rateHz);
    std::string out(buf, n > 0 ? std::min<size_t>(n, sizeof buf - 1) : 0);
    if (readout_s(s) > 0) {
        std::snprintf(buf, sizeof buf, ", readout %.1f ms%s", readout_s(s) * 1e3, s.readoutMs > 0 ? " (override)" : "");
        out += buf;
    } else {
        out += ", readout unknown (no rolling-shutter correction)";
    }
    std::string why;
    const double f = focal_px(s, &why);
    if (f > 0) {
        std::snprintf(buf, sizeof buf, ", %.1f mm%s = %.0f px", s.focalMm > 0 ? s.focalMm : h.focalMm, s.focalMm > 0 ? " (override)" : "", f);
        out += buf;
        if (s.enable) {
            const double manual = std::clamp(s.zoom, 0.1, 10.0);
            double zlo, zmean, zhi;
            zoom_stats(s, zlo, zmean, zhi);
            if (s.zoomMode == 1 && s.autoZoom) std::snprintf(buf, sizeof buf, ", zoom %.3f to %.3f (dynamic, mean %.3f)", zlo * manual, zhi * manual, zmean * manual);
            else std::snprintf(buf, sizeof buf, ", zoom %.3f", zhi * manual);
            out += buf;
            if (int n = limited_frames(s)) out += " (at the limit: stabilisation reduced on " + std::to_string(n) + " frames)";
        }
    }
    if (ends_early(s)) {
        out += " | gyro data ends at frame " + std::to_string(frames());
        out += s.clipFrames > 0 ? " of " + std::to_string(s.clipFrames) : std::string(" (before the end of the clip)");
        out += ": stabilisation fades out over the second before it";
    }
    if (noData > 0) out += " | no gyro data on " + std::to_string(noData) + " of " + std::to_string(frames()) + " frames: not stabilised there";
    if (lostSamples > 0) out += " | " + std::to_string(lostSamples) + " gyro samples missing (bridged)";
    {
        // The sync in use and where it comes from.
        const double e = exposure_s(0);
        if (delay_from_exposure()) {
            if (exposureByFrame.empty()) std::snprintf(buf, sizeof buf, " | sync %.1f ms = readout/2 + exposure/2 (1/%.0f s)", delay_s(0) * 1e3, 1.0 / e);
            else std::snprintf(buf, sizeof buf, " | sync %.1f ms at the start = readout/2 + exposure/2 (the exposure changes in the clip)", delay_s(0) * 1e3);
        } else {
            std::snprintf(buf, sizeof buf, " | sync %.1f ms from the gyro data (%s unknown)", h.markDelayS * 1e3, h.readoutS > 0 ? "exposure time" : "readout time");
        }
        out += buf;
        if (s.syncMs != 0) {
            std::snprintf(buf, sizeof buf, " %+.1f ms offset", s.syncMs);
            out += buf;
        }
    }
    if (lateMarks > 0) out += " | " + std::to_string(lateMarks) + " late frame marks corrected";
    if (droppedFrames > 0) out += " | " + std::to_string(droppedFrames) + " frames dropped by the camera";
    {
        int first, last;
        used_range(s, first, last);
        if (first == 0 && last == frames() - 1) out += " | range: whole clip";
        else out += " | range: frames " + std::to_string(first + 1) + " to " + std::to_string(last + 1) + " of " + std::to_string(frames());
        if (!s.rangeNote.empty()) out += " (" + s.rangeNote + ")";
    }
    if (!s.enable) return "Off (switched off) | " + out;
    if (!(f > 0)) return "Off: " + why + " | " + out;
    return "On | " + out;
}

// ---- lookup ----

std::string find_fpg(const std::string& dngPath, std::string& why) {
    struct Hit { std::string path, why; std::chrono::steady_clock::time_point at; };
    static std::mutex m;
    static std::map<std::string, Hit> cache;
    size_t slash = dngPath.find_last_of("/\\");
    if (slash == std::string::npos) { why = "no folder in the source path"; return ""; }
    const std::string dir = dngPath.substr(0, slash);
    std::string dngStem = upper(stem(dngPath.substr(slash + 1)));
    // Every frame of a sequence shares one lookup (the frame number is not part of the key).
    std::string key = dngStem;
    while (!key.empty() && std::isdigit(static_cast<unsigned char>(key.back()))) key.pop_back();
    key = upper(dir) + "|" + key;
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> l(m);
        auto it = cache.find(key);
        if (it != cache.end() && now - it->second.at < std::chrono::seconds(2)) { why = it->second.why; return it->second.path; }
    }
    std::vector<std::string> names;
    for (const auto& f : os::list_matching(dir + os::kSep, ".FPG")) names.push_back(f.name);
    std::string found, reason;
    size_t best = 0;
    for (const auto& n : names) {
        std::string st = upper(stem(n));
        bool match = dngStem == st || (dngStem.size() > st.size() && dngStem.compare(0, st.size(), st) == 0 && dngStem[st.size()] == '_');
        if (match && st.size() >= best) { best = st.size(); found = n; }
    }
    if (found.empty()) {
        const std::string folder = upper(file_name(dir));
        for (const auto& n : names)
            if (upper(stem(n)) == folder) found = n;
    }
    if (found.empty() && names.size() == 1) found = names[0];
    if (found.empty())
        reason = names.empty() ? "no .FPG gyro file next to the DNGs" : std::to_string(names.size()) + " .FPG files in the folder, none named like the clip";
    else
        found = dir + os::kSep + found;
    {
        std::lock_guard<std::mutex> l(m);
        if (cache.size() > 256) cache.clear();
        cache[key] = {found, reason, now};
    }
    why = reason;
    return found;
}

// Number of the clip's DNG files (same folder, same name up to the frame number), 0 if unknown.
static long long count_sequence(const std::string& dngPath) {
    struct Hit { long long count; std::chrono::steady_clock::time_point at; };
    static std::mutex m;
    static std::map<std::string, Hit> cache;
    std::string prefix, suffix;
    long long number = 0;
    int digits = 0;
    if (!split_sequence(dngPath, prefix, number, digits, suffix)) return 0;
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> l(m);
        auto it = cache.find(prefix);
        if (it != cache.end() && now - it->second.at < std::chrono::seconds(10)) return it->second.count;
    }
    const long long count = static_cast<long long>(os::list_matching(prefix, suffix).size());
    {
        std::lock_guard<std::mutex> l(m);
        if (cache.size() > 256) cache.clear();
        cache[prefix] = {count, now};
    }
    return count;
}

std::shared_ptr<const GyroClip> load_gyro(const std::string& fpgPath, std::string& error, const std::string& dngPath) {
    struct Entry { uint64_t size = 0, time = 0; std::shared_ptr<const GyroClip> clip; std::string error; };
    static std::mutex m;
    static std::map<std::string, Entry> cache;
    os::FileInfo st;
    if (!os::stat_file(fpgPath, st)) { error = "cannot open " + file_name(fpgPath); return nullptr; }
    const uint64_t size = st.size, time = st.time;
    std::lock_guard<std::mutex> l(m);
    auto it = cache.find(fpgPath);
    if (it != cache.end() && it->second.size == size && it->second.time == time) {
        error = it->second.error;
        return it->second.clip;
    }
    Entry e;
    e.size = size; e.time = time;
    std::vector<uint8_t> data;
    std::string err;
    if (read_file(fpgPath, data, err)) {
        auto clip = GyroClip::parse(data.data(), data.size(), err);
        if (clip) {
            // The camera's sidecar files carry no exposure time: take it from the clip's DNG.
            if (!(clip->h.exposureS > 0) && !dngPath.empty()) clip->h.exposureS = dng_exposure(dngPath);
            clip->path = fpgPath;
            clip->name = file_name(fpgPath);
            e.clip = clip;
        } else {
            err = file_name(fpgPath) + ": " + err;
        }
    }
    if (!e.clip) e.error = err;
    if (cache.size() >= 8) cache.clear();
    cache[fpgPath] = e;
    error = e.error;
    return e.clip;
}

std::shared_ptr<const GyroClip> gyro_for_frame(const std::string& dngPath, const std::string& gyroFile, long long frame,
                                               int dngW, int dngH, const StabSettings& s, StabParams& warp, std::string& status,
                                               bool wait) {
    warp.on = 0;
    std::string why;
    std::shared_ptr<const GyroClip> clip;
    StabSettings t = s;
    if (gyroFile.empty()) {
        GyroLoad load = wait ? inframe_gyro_wait(dngPath) : inframe_gyro(dngPath);
        if (load.loading) {
            status = "Loading gyro " + std::to_string(load.percent) + " % (" + std::to_string(load.frames) + " frames) | not stabilised until it is ready";
            return nullptr;
        }
        clip = load.clip;
        why = load.note;
    }
    if (clip) {
        t.clipFrames = clip->frames();     // the track covers the clip; frames without data are marked in it
    } else {
        const std::string unusable = why;
        std::string file = gyroFile.empty() ? find_fpg(dngPath, why) : gyroFile;
        if (file.empty()) {
            status = unusable.empty() ? "Off: no gyro data in the DNGs, " + why : "Off: the gyro data in the DNGs cannot be used: " + unusable;
            return nullptr;
        }
        clip = load_gyro(file, why, dngPath);
        if (!clip) { status = "Off: " + why; return nullptr; }
        if (t.clipFrames <= 0) t.clipFrames = count_sequence(dngPath);
    }
    if (dngW > 0 && dngH > 0 && (dngW != clip->h.rasterW || dngH != clip->h.rasterH)) {
        status = "Off: " + (clip->inFrame ? std::string("the gyro data") : clip->name) + " is for a " + std::to_string(clip->h.rasterW) + "x" +
                 std::to_string(clip->h.rasterH) + " raster, the DNG is " + std::to_string(dngW) + "x" + std::to_string(dngH);
        return clip;
    }
    clip->warp(frame, t, warp, why);
    status = clip->status(t);
    return clip;
}

}  // namespace sfp
