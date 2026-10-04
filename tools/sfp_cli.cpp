// Sigma fp RAW command-line tool: develop one frame to a 16-bit TIFF, or benchmark
// real-time playback of a sequence (cache + read-ahead + GPU develop into a GPU
// image, as in Resolve's CUDA render path).
//   sfp_cli develop IN.DNG OUT.tif [key=value ...]
//   sfp_cli play FIRST.DNG COUNT [fps=50] [key=value ...]
// keys: phase=auto|off|<value>  quality=0|1  exposure= wb=asshot|custom temp= tint=
//       cs=709|p3|2020|dwg|ap0|ap1  gamma=lin|22|24|709|srgb|di|cct  sharp= sat= contrast=
//       hl= sh= mid= boost= lift= gain= hr=0|1 width= height= fit=fit|fill|stretch|native
//       device=1 (develop only: render into a GPU image like a CUDA host, then download)
//       clip=<DNG frames of the clip> (gyro only; found by itself for a DNG path)
//       range=<first>-<last> (DNG numbers the smoothing and zoom are based on) zoommode=fixed|dynamic zoomsmooth=<s>
//       marks=raw (each frame's time from its own mark, as before v1.3.1; for measurements)
//       stab=0|1 smooth=<s> rs=0..1 sync=<ms> focal=<mm> readout=<ms> autozoom=0|1 maxzoom= zoom= gyro=<file.FPG>
//   sfp_cli gyro FILE.FPG|CLIP_FRAME.DNG [frame=N] [path=1] [stab keys] [pt=x,y ...]
//     prints the file's header and, for frame N, the stabilisation warp as JSON; each pt is a
//     raster position of the stabilised image, reported with the position it is read from.
//     Given a DNG path, it reports the gyro data the plug-in picks for that clip (in-frame blocks, else the .FPG).
//   sfp_cli selftest [FILE.FPG]   develops a synthetic frame (no footage needed) and checks the result; exit code 0 = pass.
//   sfp_cli fpg2 FRAME.DNG       prints the frame's in-frame gyro block as JSON.
//   sfp_cli fpg2scan FIRST.DNG [threads=N] [fixed=1] [direct=1]   reads the block region of every frame of the clip and prints the time taken.
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "../src/develop.h"
#include "../src/fp_chart_matrix.h"
#include "../src/gyro.h"

using namespace sfp;
using clk = std::chrono::steady_clock;

static StabSettings gStab;
static std::string gGyroFile;
static bool gDevice = false;   // develop into a GPU image (the path a CUDA host uses), then download

static bool apply(RawSettings& s, const std::string& kv, int& w, int& h, double& fps) {
    auto eq = kv.find('=');
    if (eq == std::string::npos) return false;
    std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
    double d = std::atof(v.c_str());
    if (k == "phase") s.rowPhase = v == "auto" ? -0.125 : v == "off" ? 0.0 : d; else if (k == "quality") s.decodeQuality = static_cast<int>(d);
    else if (k == "dezig" || k == "aa") s.deZigzag = d;
    else if (k == "hot") s.hotPixels = d != 0;
    else if (k == "exposure") s.exposure = d;
    else if (k == "wb") s.whiteBalance = v == "custom" ? WhiteBalance::Custom : WhiteBalance::AsShot;
    else if (k == "temp") s.colorTemp = d;
    else if (k == "tint") s.tint = d;
    else if (k == "cam") s.cameraMatrix = v == "chart" ? CameraMatrix::ChartFitted : CameraMatrix::AsRecorded;
    else if (k == "cs") s.colorSpace = v == "p3" ? Primaries::P3D65 : v == "2020" ? Primaries::Rec2020 : v == "dwg" ? Primaries::DaVinciWideGamut
                                     : v == "awg3" ? Primaries::ArriWideGamut3
                                     : v == "ap0" ? Primaries::ACES_AP0 : v == "ap1" ? Primaries::ACES_AP1 : Primaries::Rec709;
    else if (k == "gamma") s.gamma = v == "lin" ? Gamma::Linear : v == "22" ? Gamma::Gamma22 : v == "709" ? Gamma::Rec709 : v == "srgb" ? Gamma::sRGB
                                   : v == "di" ? Gamma::DaVinciIntermediate : v == "cct" ? Gamma::ACEScct : Gamma::Gamma24;
    else if (k == "sharp") s.sharpness = d;
    else if (k == "sat") s.saturation = d;
    else if (k == "contrast") s.contrast = d;
    else if (k == "hl") s.highlights = d;
    else if (k == "sh") s.shadows = d;
    else if (k == "mid") s.midtones = d;
    else if (k == "boost") s.colorBoost = d;
    else if (k == "lift") s.lift = d;
    else if (k == "gain") s.gain = d;
    else if (k == "hr") s.highlightRecovery = d != 0;
    else if (k == "width") w = static_cast<int>(d);
    else if (k == "height") h = static_cast<int>(d);
    else if (k == "fps") fps = d;
    else if (k == "stab") gStab.enable = d != 0;
    else if (k == "smooth") gStab.smoothness = d;
    else if (k == "rs") gStab.rollingShutter = d;
    else if (k == "sync") gStab.syncMs = d;
    else if (k == "focal") gStab.focalMm = d;
    else if (k == "readout") gStab.readoutMs = d;
    else if (k == "marks") gStab.rawMarks = v == "raw";
    else if (k == "autozoom") gStab.autoZoom = d != 0;
    else if (k == "zoom") gStab.zoom = d;
    else if (k == "maxzoom") gStab.maxZoom = d;
    else if (k == "clip") gStab.clipFrames = static_cast<long long>(d);
    else if (k == "range") {
        // range=A-B: DNG numbers (1-based, inclusive); B = 0 or missing: to the end
        long long a = 1, b = 0;
        std::sscanf(v.c_str(), "%lld-%lld", &a, &b);
        gStab.rangeFirst = a - 1;
        gStab.rangeLast = b > 0 ? b - 1 : 1000000000LL;
        gStab.rangeNote = "manual";
    }
    else if (k == "zoommode") gStab.zoomMode = v == "dynamic" ? 1 : 0;
    else if (k == "zoomsmooth") gStab.zoomSmooth = d;
    else if (k == "device") gDevice = d != 0;
    else if (k == "gyro") gGyroFile = v;
    else if (k == "fit") s.fit = v == "fill" ? Fit::Fill : v == "stretch" ? Fit::Stretch : v == "native" ? Fit::Native : Fit::Fit;
    else return false;
    return true;
}

static bool is_dng_name(const std::string& p) {
    if (p.size() < 4) return false;
    std::string e = p.substr(p.size() - 4);
    for (auto& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return e == ".dng";
}

static void write_tiff(const char* path, const std::vector<float>& rgba, int w, int h) {
    // Uncompressed 16-bit RGB, top-down (OFX image is bottom-up: flip).
    std::vector<uint16_t> px(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                float v = rgba[(static_cast<size_t>(h - 1 - y) * w + x) * 4 + c];
                px[(static_cast<size_t>(y) * w + x) * 3 + c] = static_cast<uint16_t>(std::lround(std::fmin(std::fmax(v, 0.f), 1.f) * 65535.f));
            }
    FILE* f = std::fopen(path, "wb");
    auto u16 = [&](uint16_t v) { std::fwrite(&v, 2, 1, f); };
    auto u32 = [&](uint32_t v) { std::fwrite(&v, 4, 1, f); };
    const uint32_t dataOff = 8, dataBytes = static_cast<uint32_t>(px.size() * 2), ifd = dataOff + dataBytes, bpsOff = ifd + 2 + 10 * 12 + 4;
    std::fwrite("II*\0", 1, 4, f); u32(ifd);
    std::fwrite(px.data(), 2, px.size(), f);
    u16(10);
    auto ent = [&](uint16_t t, uint16_t ty, uint32_t n, uint32_t v) { u16(t); u16(ty); u32(n); u32(v); };
    ent(256, 4, 1, w); ent(257, 4, 1, h); ent(258, 3, 3, bpsOff); ent(259, 3, 1, 1); ent(262, 3, 1, 2);
    ent(273, 4, 1, dataOff); ent(277, 3, 1, 3); ent(278, 4, 1, h); ent(279, 4, 1, dataBytes); ent(284, 3, 1, 1);
    u32(0); u16(16); u16(16); u16(16);
    std::fclose(f);
}

// Header and one frame's warp of a gyro file, as JSON.
static int gyro_report(int argc, char** argv) {
    RawSettings s;
    int w = 0, h = 0;
    double fps = 0;
    long long frame = 0;
    bool path = false;
    std::vector<std::pair<double, double>> pts;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        double x, y;
        if (a.rfind("frame=", 0) == 0) frame = std::atoll(a.c_str() + 6);
        else if (a == "path=1") path = true;
        else if (std::sscanf(a.c_str(), "pt=%lf,%lf", &x, &y) == 2) pts.push_back({x, y});
        else if (!apply(s, a, w, h, fps)) { std::printf("bad option %s\n", argv[i]); return 2; }
    }
    std::string err, file = argv[2];
    auto json = [](const std::string& t) {
        std::string o;
        for (char c : t) {
            if (c == '\\' || c == '"') o += '\\';
            o += c;
        }
        return o;
    };
    // A DNG path: the gyro data the plug-in uses for that clip (in-frame blocks first, then the sidecar).
    std::shared_ptr<const GyroClip> clip;
    GyroLoad load;
    if (file.size() > 4 && is_dng_name(file)) {
        std::string dng = file, prefix, suffix;
        long long number = 0;
        int digits = 0;
        load = inframe_gyro_wait(dng);
        clip = load.clip;
        if (clip) {
            gStab.clipFrames = clip->frames();
        } else {
            file = find_fpg(dng, err);
            if (!file.empty() && !gStab.clipFrames && split_sequence(dng, prefix, number, digits, suffix)) {
                // The clip's DNG count, as the plug-in gets it.
                for (long long n = 1; n < 1000000; ++n) {
                    FILE* fp = std::fopen(sequence_path(prefix, n, digits, suffix).c_str(), "rb");
                    if (!fp) break;
                    std::fclose(fp);
                    gStab.clipFrames = n;
                }
            }
            if (file.empty()) { std::printf("{\"error\": \"no gyro data in the DNGs, %s\", \"scan_ms\": %.1f}\n", json(load.note.empty() ? err : load.note).c_str(), load.scanMs); return 1; }
        }
    }
    if (!clip) clip = load_gyro(file, err, argv[2] == file ? std::string() : std::string(argv[2]));
    if (!clip) { std::printf("{\"error\": \"%s\", \"file\": \"%s\"}\n", json(err).c_str(), json(file).c_str()); return 1; }
    std::printf("{\"source\": \"%s\", \"scan_ms\": %.1f, \"from_cache\": %d, \"blocks\": %d, \"no_data\": %d, \"lost_samples\": %lld,\n ",
                clip->inFrame ? "in-frame" : "fpg", load.scanMs, load.fromCache ? 1 : 0, clip->blocks, clip->noData, clip->lostSamples);
    if (clip->inFrame) file.clear();
    const GyroHeader& g = clip->h;
    std::printf("\"file\": \"%s\",\n ", json(file).c_str());
    std::printf("\"version\": %d, \"frames\": %u, \"samples\": %u, \"rate_hz\": %.6f, \"lsb_per_dps\": %.6f,\n", g.version, g.frameCount, g.sampleCount, g.rateHz, g.lsbPerDps);
    std::printf(" \"raster\": [%d, %d], \"active\": [%d, %d, %d, %d], \"fps\": %.6f, \"readout_us\": %.0f, \"exposure_us\": %.0f,\n",
                g.rasterW, g.rasterH, g.activeX, g.activeY, g.activeW, g.activeH, g.fps, g.readoutS * 1e6, g.exposureS * 1e6);
    std::printf(" \"focal_um\": %.0f, \"mark_delay_us\": %.0f, \"axes\": [%d, %d, %d], \"flags\": %u, \"preroll\": %u, \"pitch_nm\": %.0f,\n",
                g.focalMm * 1e3, g.markDelayS * 1e6, g.axis[0], g.axis[1], g.axis[2], g.flags, g.preroll, g.pitchNm);
    std::printf(" \"resolution\": %u, \"dc_crop\": %u, \"bit_depth\": %u, \"sensor_mode\": %u,\n", g.resolution, g.dcCrop, g.bitDepth, g.sensorMode);
    StabParams p{};
    std::string why;
    clip->warp(frame, gStab, p, why);
    const bool on = p.on != 0;
    std::printf(" \"why_off\": \"%s\",\n", on ? "" : json(why).c_str());
    Quat a = clip->orientation(clip->frame_time(frame, gStab.syncMs, gStab.rawMarks)), sm = clip->smoothed(frame, gStab);
    std::printf(" \"delay_ms\": %.4f, \"delay_from_exposure\": %d, \"exposure_s\": %.8f,\n", clip->delay_s(frame) * 1e3, clip->delay_from_exposure() ? 1 : 0, clip->exposure_s(frame));
    std::printf(" \"dropped_frames\": %d, \"late_marks\": %d, \"typical_mark_lateness_ms\": %.4f,\n", clip->droppedFrames, clip->lateMarks, clip->typicalLateMs);
    std::printf(" \"status\": \"%s\", \"frame\": %lld, \"time\": %.9f, \"on\": %d, \"focal_px\": %.6f, \"zoom\": %.6f, \"auto_zoom\": %.6f,\n",
                clip->status(gStab).c_str(), frame, clip->frame_time(frame, gStab.syncMs, gStab.rawMarks), on ? 1 : 0, clip->focal_px(gStab),
                on ? 1.0 / p.invZoom : 1.0, clip->auto_zoom(gStab));
    std::printf(" \"ends_early\": %d, \"clip_frames\": %lld, \"frame_zoom\": %.6f,\n", clip->ends_early(gStab) ? 1 : 0, gStab.clipFrames, clip->frame_zoom(frame, gStab));
    std::printf(" \"limited_frames\": %d,\n", clip->limited_frames(gStab));
    {
        int first, last;
        double zlo, zmean, zhi;
        clip->used_range(gStab, first, last);
        clip->zoom_stats(gStab, zlo, zmean, zhi);
        bool ps, pe;
        clip->range_ends(gStab, ps, pe);
        std::printf(" \"range\": [%d, %d], \"smoothing_past_range\": [%d, %d], \"zoom_min\": %.6f, \"zoom_mean\": %.6f, \"zoom_max\": %.6f,\n",
                    first + 1, last + 1, ps ? 1 : 0, pe ? 1 : 0, zlo, zmean, zhi);
    }
    std::printf(" \"orientation\": [%.12f, %.12f, %.12f, %.12f], \"smoothed\": [%.12f, %.12f, %.12f, %.12f],\n", a.w, a.x, a.y, a.z, sm.w, sm.x, sm.y, sm.z);
    if (path) {
        // Every frame: readout-middle time, actual and virtual-camera orientation (w, x, y, z), automatic zoom, has gyro data,
        // the mark as recorded and on the fitted cadence.
        std::printf(" \"path\": [");
        for (int i = 0; i < clip->frames(); ++i) {
            Quat q = clip->orientation(clip->frame_time(i, gStab.syncMs, gStab.rawMarks)), v = clip->smoothed(i, gStab);
            std::printf("%s[%.9f, %.12f, %.12f, %.12f, %.12f, %.12f, %.12f, %.12f, %.12f, %.6f, %d, %.9f, %.9f]", i ? ",\n  " : "",
                        clip->frame_time(i, gStab.syncMs, gStab.rawMarks),
                        q.w, q.x, q.y, q.z, v.w, v.x, v.y, v.z, clip->frame_zoom(i, gStab), clip->has_data(i) ? 1 : 0,
                        clip->mark_recorded(i), clip->mark_regular(i));
        }
        std::printf("],\n");
    }
    std::printf(" \"points\": [");
    for (size_t i = 0; i < pts.size(); ++i) {
        float sx = static_cast<float>(pts[i].first), sy = static_cast<float>(pts[i].second);
        if (on) stab_map(p, sx, sy, &sx, &sy);
        std::printf("%s[%.4f, %.4f, %.4f, %.4f]", i ? ", " : "", pts[i].first, pts[i].second, sx, sy);
    }
    std::printf("]}\n");
    return 0;
}

// The in-frame gyro block of one DNG, as JSON.
static int block_report(const char* path) {
    GyroBlock b;
    std::string why;
    if (!read_fpg2(path, b, &why)) { std::printf("{\"present\": 0, \"why\": \"%s\"}\n", why.c_str()); return 1; }
    std::printf("{\"present\": 1, \"frame\": %u, \"clock_us\": %u, \"n\": %zu, \"ring_index\": %u, \"flags\": %u, \"lost\": %u, \"take\": %u,\n"
                " \"readout_us\": %u, \"mark_delay_us\": %d, \"sensor_mode\": %u, \"resolution\": %u, \"dc_crop\": %u, \"bit_depth\": %u,\n"
                " \"axes\": [%d, %d, %d], \"lsb_per_dps\": %u, \"samples\": [",
                b.frame, b.clockUs, b.samples.size() / 3, b.ringIndex, b.flags, b.lost, b.take, b.readoutUs, b.markDelayUs, b.sensorMode,
                b.resolution, b.dcCrop, b.bitDepth, b.axis[0], b.axis[1], b.axis[2], b.lsb);
    for (size_t i = 0; i < b.samples.size(); ++i) std::printf("%s%d", i ? ", " : "", b.samples[i]);
    std::printf("]}\n");
    return 0;
}

// Develops a synthetic frame on this machine and checks the picture: finite values, a neutral
// grey stays neutral, half and full resolution agree, and an identity gyro warp changes nothing.
// Camera Matrix = Chart-Fitted on a Sigma fp Color Mode OFF pair (one body's ColorMatrix1/2, Standard A / D65).
// Each check comes with a control that must fail, so a check that cannot see an error is caught too.
static bool colour_selftest() {
    DngInfo d;
    const double off1[9] = {1.5596, -0.7398, -0.2399, -1.1733, 2.271, 0.0503, 0.0835, -0.3064, 1.588};
    const double off2[9] = {1.6, -0.759, -0.2461, -0.9692, 1.876, 0.0415, 0.0345, -0.1267, 0.6567};
    for (int i = 0; i < 9; ++i) { d.cm1[i] = off1[i]; d.cm2[i] = off2[i]; }
    d.hasCM1 = d.hasCM2 = true;
    d.illum1 = 17; d.illum2 = 21;
    auto neutral_for = [&](const double* cm, double x, double y) {
        const double w[3] = {x / y, 1.0, (1 - x - y) / y};
        for (int i = 0; i < 3; ++i) d.neutral[i] = cm[3 * i] * w[0] + cm[3 * i + 1] * w[1] + cm[3 * i + 2] * w[2];
        for (int i : {0, 2, 1}) d.neutral[i] /= d.neutral[1];
    };
    auto maxdiff = [](const float* a, const double* b) {
        double m = 0;
        for (int i = 0; i < 9; ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
        return m;
    };
    bool ok = true;
    // 1. White balance and Temp/Tint unchanged, at both calibration whites.
    for (int k = 0; k < 2; ++k) {
        neutral_for(k ? off2 : off1, kFpChartWhite[k][0], kFpChartWhite[k][1]);
        ColorSetup a = color_setup(d, true, 0, 0, Primaries::Rec709, CameraMatrix::AsRecorded);
        ColorSetup c = color_setup(d, true, 0, 0, Primaries::Rec709, CameraMatrix::ChartFitted);
        double dt = std::fabs(a.temp - c.temp), dn = std::fabs(a.tint - c.tint);
        bool pass = dt < 2 && dn < 0.2 && std::fabs(a.wb[0] - c.wb[0]) < 1e-5 && std::fabs(a.wb[2] - c.wb[2]) < 1e-5;
        std::printf("chart matrix %s: as shot %.0f K / %.2f, chart-fitted %.0f K / %.2f  %s\n", k ? "D65 " : "StdA",
                    a.temp, a.tint, c.temp, c.tint, pass ? "ok" : "FAIL");
        ok = ok && pass;
    }
    // 2. Colour at D65 is the fit (kFpChartTo709); the recorded OFF pair is not (control).
    neutral_for(off2, 0.3127, 0.3290);
    double fit = maxdiff(color_setup(d, true, 0, 0, Primaries::Rec709, CameraMatrix::ChartFitted).matrix, kFpChartTo709);
    double rec = maxdiff(color_setup(d, true, 0, 0, Primaries::Rec709, CameraMatrix::AsRecorded).matrix, kFpChartTo709);
    std::printf("chart matrix colour: chart-fitted vs fit %.5f %s, as recorded vs fit %.3f %s\n", fit, fit < 1e-3 ? "ok" : "FAIL",
                rec, rec > 0.1 ? "(control caught)" : "(control MISSED)");
    ok = ok && fit < 1e-3 && rec > 0.1;
    // 3. Idempotent: a file that already carries the chart pair decodes the same either way.
    CmPair once = chart_fitted_matrices(recorded_matrices(d)), twice = chart_fitted_matrices(once);
    double idem = 0;
    for (int i = 0; i < 9; ++i) idem = std::fmax(idem, std::fmax(std::fabs(once.c1.m[i] - twice.c1.m[i]), std::fabs(once.c2.m[i] - twice.c2.m[i])));
    CmPair recorded = recorded_matrices(d);
    double moved = 0;
    for (int i = 0; i < 9; ++i) moved = std::fmax(moved, std::fabs(once.c2.m[i] - recorded.c2.m[i]));
    std::printf("chart matrix idempotent: %.2e %s, recorded -> chart-fitted moves %.3f %s\n", idem, idem < 1e-9 ? "ok" : "FAIL",
                moved, moved > 0.1 ? "(control caught)" : "(control MISSED)");
    ok = ok && idem < 1e-9 && moved > 0.1;
    // 4. ARRI Wide Gamut 3: luminance row against ARRI's published AWG3 -> XYZ; grey stays grey.
    ColorSetup g = color_setup(d, true, 0, 0, Primaries::ArriWideGamut3, CameraMatrix::ChartFitted);
    const double arriY[3] = {0.291954, 0.823841, -0.115795};
    double dy = 0, grey = 0;
    for (int i = 0; i < 3; ++i) {
        dy = std::fmax(dy, std::fabs(g.lum[i] - arriY[i]));
        grey = std::fmax(grey, std::fabs(g.matrix[3 * i] + g.matrix[3 * i + 1] + g.matrix[3 * i + 2] - 1.0));
    }
    std::printf("AWG3: Y row vs ARRI %.5f %s, grey %.5f %s\n", dy, dy < 1e-3 ? "ok" : "FAIL", grey, grey < 1e-3 ? "ok" : "FAIL");
    ok = ok && dy < 1e-3 && grey < 1e-3;
    return ok;
}

static int selftest(const char* fpg) {
    const int W = 640, H = 480;
    Frame f;
    f.info.width = W; f.info.height = H; f.info.bps = 12; f.info.compression = 1;
    f.info.cropW = W; f.info.cropH = H;
    f.info.white = 4095;
    for (int k = 0; k < 4; ++k) f.info.black[k] = 256;
    // XYZ (D65) -> camera = XYZ -> linear sRGB: the camera sees sRGB primaries, white is neutral.
    const double m[9] = {3.2404542, -1.5371385, -0.4985314, -0.9692660, 1.8760108, 0.0415560, 0.0556434, -0.2040259, 1.0572252};
    for (int i = 0; i < 9; ++i) f.info.cm1[i] = m[i];
    f.info.hasCM1 = true;
    f.info.illum1 = 21;
    f.raw.resize(static_cast<size_t>(W) * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            // Left half: flat grey. Right half: a soft diagonal ramp (exercises the demosaic).
            double v = x < W / 2 ? 0.18 : 0.05 + 0.4 * (x - W / 2 + y) / double(W / 2 + H);
            f.raw[static_cast<size_t>(y) * W + x] = static_cast<uint16_t>(256 + v * (4095 - 256) + 0.5);
        }
    std::printf("backend: %s\n", Developer::backend());
    auto run = [&](int quality, bool stab, uint64_t id, std::vector<float>& img) {
        RawSettings s;
        s.decodeQuality = quality;
        s.gamma = Gamma::Linear;
        if (stab) {
            s.stab.on = 1; s.stab.focal = 1000; s.stab.cx = W / 2.f; s.stab.cy = H / 2.f; s.stab.rows = H; s.stab.invZoom = 1;
            for (int k = 0; k < STAB_KNOTS; ++k) s.stab.rot[9 * k] = s.stab.rot[9 * k + 4] = s.stab.rot[9 * k + 8] = 1;
        }
        f.id = id;
        img.assign(static_cast<size_t>(W) * H * 4, -1.f);
        Target t;
        t.host = img.data(); t.width = W; t.height = H; t.rowBytes = W * 16; t.rodW = W; t.rodH = H;
        std::string err;
        DevelopTiming tm;
        if (!Developer::get().develop(f, s, nullptr, t, err, &tm)) { std::printf("FAIL develop: %s\n", err.c_str()); return false; }
        std::printf("quality %d%s: %.1f ms\n", quality, stab ? " + warp" : "", tm.gpuMs);
        return true;
    };
    std::vector<float> full, half, warped;
    if (!run(0, false, 1, full) || !run(1, false, 2, half) || !run(0, true, 3, warped)) return 1;
    int bad = 0;
    double grey[3] = {0, 0, 0}, dHalf = 0, dWarp = 0;
    long long n = 0;
    for (int y = 16; y < H - 16; ++y)
        for (int x = 16; x < W - 16; ++x) {
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            for (int c = 0; c < 4; ++c)
                if (!std::isfinite(full[i + c]) || !std::isfinite(half[i + c]) || !std::isfinite(warped[i + c])) ++bad;
            for (int c = 0; c < 3; ++c) {
                dHalf = std::fmax(dHalf, std::fabs(full[i + c] - half[i + c]));
                dWarp = std::fmax(dWarp, std::fabs(full[i + c] - warped[i + c]));
            }
            if (x < W / 2 - 16) { for (int c = 0; c < 3; ++c) grey[c] += full[i + c]; ++n; }
        }
    for (double& g : grey) g /= static_cast<double>(n);
    std::printf("grey %.4f %.4f %.4f, half vs full %.4f, warp vs none %.5f, non-finite %d\n", grey[0], grey[1], grey[2], dHalf, dWarp, bad);
    bool ok = bad == 0 && grey[1] > 0.05 && grey[1] < 0.6 && std::fabs(grey[0] - grey[1]) < 0.01 * grey[1] + 1e-4 &&
              std::fabs(grey[2] - grey[1]) < 0.01 * grey[1] + 1e-4 && dHalf < 0.05 && dWarp < 1e-3;
    if (fpg) {
        std::string err;
        auto clip = load_gyro(fpg, err, "");
        if (!clip) { std::printf("FAIL gyro file: %s\n", err.c_str()); ok = false; }
        else {
            std::printf("gyro file: %lld frames, %.2f Hz\n", static_cast<long long>(clip->frames()), clip->h.rateHz);
            if (clip->frames() < 1) ok = false;
        }
    }
    ok = colour_selftest() && ok;
    std::puts(ok ? "PASS" : "FAIL");
    std::fflush(stdout);
    std::_Exit(ok ? 0 : 1);
}

int main(int argc, char** argv) {
    if (argc >= 2 && !std::strcmp(argv[1], "selftest")) return selftest(argc >= 3 ? argv[2] : nullptr);
    if (argc >= 3 && !std::strcmp(argv[1], "gyro")) { int r = gyro_report(argc, argv); std::fflush(stdout); gyro_shutdown(); return r; }
    if (argc >= 3 && !std::strcmp(argv[1], "fpg2")) return block_report(argv[2]);
    if (argc >= 3 && !std::strcmp(argv[1], "fpg2scan")) {
        long long files = 0, found = 0;
        int threads = 0;
        bool fixedOnly = false, direct = false;
        for (int i = 3; i < argc; ++i) {
            if (!std::strncmp(argv[i], "threads=", 8)) threads = std::atoi(argv[i] + 8);
            else if (!std::strcmp(argv[i], "fixed=1")) fixedOnly = true;
            else if (!std::strcmp(argv[i], "direct=1")) fixedOnly = direct = true;
        }
        double ms = scan_blocks(argv[2], files, found, threads, fixedOnly, direct);
        std::printf("{\"files\": %lld, \"blocks\": %lld, \"ms\": %.1f, \"ms_per_1000\": %.1f}\n", files, found, ms, files ? ms * 1000.0 / files : 0.0);
        return 0;
    }
    if (argc < 4) { std::puts("usage: sfp_cli develop IN.DNG OUT.tif [k=v..] | play FIRST.DNG COUNT [k=v..] | gyro FILE.FPG [frame=N] [k=v..] [pt=x,y ..] | selftest [FILE.FPG]"); return 2; }
    std::string mode = argv[1];
    RawSettings s;
    int w = 0, h = 0;
    double fps = 50;
    for (int i = 4; i < argc; ++i)
        if (!apply(s, argv[i], w, h, fps)) { std::printf("bad option %s\n", argv[i]); return 2; }
    std::string err;
    // Gyro stabilisation, as in the plug-in: DNG number N of a clip is gyro frame N - 1.
    auto stabilise = [&](const std::string& path, const Frame& f, bool print) {
        std::string prefix, suffix, status;
        long long number = 0;
        int digits = 0;
        bool seq = split_sequence(path, prefix, number, digits, suffix);
        gyro_for_frame(path, gGyroFile, seq ? number - 1 : 0, f.info.width, f.info.height, gStab, s.stab, status, true);
        if (print) std::printf("gyro: %s\n", status.c_str());
    };
    if (mode == "develop") {
        auto t0 = clk::now();
        auto f = FrameCache::get().fetch(argv[2], err);
        if (!f) { std::printf("ERR %s\n", err.c_str()); return 1; }
        double fetchMs = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        if (!w) { w = f->info.cropW; h = f->info.cropH; }
        stabilise(argv[2], *f, true);
        std::vector<float> img(static_cast<size_t>(w) * h * 4);
        Target t;
        t.host = img.data(); t.width = w; t.height = h; t.rowBytes = w * 16; t.rodW = w; t.rodH = h;
        DevelopTiming tm;
        if (gDevice) {
            auto& cu = cuda();
            if (!cu.ok) { std::printf("ERR %s\n", cu.error.c_str()); return 1; }
            CUdevice dev; CUcontext ctx; CUstream stream; CUdeviceptr out;
            const size_t bytes = img.size() * sizeof(float);
            cu.cuDeviceGet(&dev, 0);
            cu.cuDevicePrimaryCtxRetain(&ctx, dev);
            cu.cuCtxPushCurrent(ctx);
            cu.cuMemAlloc(&out, bytes);
            cu.cuStreamCreate(&stream, 0);
            Target td = t;
            td.host = nullptr; td.device = out;
            if (!Developer::get().develop(*f, s, stream, td, err)) { std::printf("ERR %s\n", err.c_str()); return 1; }
            cu.cuMemcpyDtoHAsync(img.data(), out, bytes, stream);
            cu.cuStreamSynchronize(stream);
            write_tiff(argv[3], img, w, h);
            std::printf("%s -> %s (GPU image)\n", argv[2], argv[3]);
            std::fflush(stdout);
            std::_Exit(0);
        }
        for (int rep = 0; rep < 3; ++rep) {
            if (!Developer::get().develop(*f, s, nullptr, t, err, &tm)) { std::printf("ERR %s\n", err.c_str()); return 1; }
            if (!rep) std::printf("backend: %s\n", Developer::backend());
            std::printf("develop %s: upload %.2f ms, total %.2f ms%s\n", rep ? "(again)" : "", tm.uploadMs, tm.gpuMs, tm.demosaicReused ? " (demosaic reused)" : "");
        }
        write_tiff(argv[3], img, w, h);
        std::printf("%s: %dx%d comp %d, read %.1f ms, decode %.1f ms (fetch %.1f ms), phase %.3f -> %s\n", argv[2], f->info.width, f->info.height,
                    f->info.compression, f->readMs, f->decodeMs, fetchMs, effective_row_phase(s, f->info), argv[3]);
        return 0;
    }
    if (mode == "play") {
        std::string prefix, suffix;
        long long first;
        int digits;
        if (!split_sequence(argv[2], prefix, first, digits, suffix)) { std::puts("not a numbered sequence"); return 2; }
        const int count = std::atoi(argv[3]);
        auto& cu = cuda();
        if (!cu.ok) { std::printf("ERR %s\n", cu.error.c_str()); return 1; }
        CUdevice dev; CUcontext ctx;
        cu.cuDeviceGet(&dev, 0);
        cu.cuDevicePrimaryCtxRetain(&ctx, dev);
        cu.cuCtxPushCurrent(ctx);
        char name[128] = {};
        cu.cuDeviceGetName(name, sizeof name, dev);
        if (!w) { w = 3840; h = 2160; }
        CUdeviceptr out;
        cu.cuMemAlloc(&out, static_cast<size_t>(w) * h * 16);
        CUstream stream;
        cu.cuStreamCreate(&stream, 0);
        Target t;
        t.device = out; t.width = w; t.height = h; t.rowBytes = w * 16; t.rodW = w; t.rodH = h;
        const double period = 1000.0 / fps;
        std::vector<double> lat;
        int late = 0;
        auto start = clk::now();
        for (int i = 0; i < count; ++i) {
            auto due = start + std::chrono::microseconds(static_cast<long long>(i * period * 1000));
            std::this_thread::sleep_until(due);
            auto t0 = clk::now();
            std::string path = sequence_path(prefix, first + i, digits, suffix);
            std::vector<std::string> ahead;
            for (int k = 1; k <= 32; ++k) ahead.push_back(sequence_path(prefix, first + i + k, digits, suffix));
            auto f = FrameCache::get().fetch(path, err);
            FrameCache::get().prefetch(ahead);
            if (!f) { std::printf("ERR %s: %s\n", path.c_str(), err.c_str()); return 1; }
            stabilise(path, *f, false);
            if (!Developer::get().develop(*f, s, stream, t, err)) { std::printf("ERR %s\n", err.c_str()); return 1; }
            cu.cuStreamSynchronize(stream);
            double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            lat.push_back(ms);
            if (ms > period) ++late;   // processing longer than one frame period
        }
        double wall = std::chrono::duration<double>(clk::now() - start).count();
        std::vector<double> sorted = lat;
        std::sort(sorted.begin(), sorted.end());
        auto st = FrameCache::get().stats();
        std::printf("%s | %d frames at %.2f fps target into %dx%d: achieved %.2f fps, latency median %.2f ms, p95 %.2f ms, max %.2f ms, late %d\n",
                    name, count, fps, w, h, count / wall, sorted[sorted.size() / 2], sorted[sorted.size() * 95 / 100], sorted.back(), late);
        std::printf("cache hits %llu misses %llu waits %llu prefetched %llu failed %llu, %zu MB\n",
                    (unsigned long long)st.hits, (unsigned long long)st.misses, (unsigned long long)st.waits,
                    (unsigned long long)st.prefetched, (unsigned long long)st.failed, st.bytes >> 20);
        std::fflush(stdout);
        std::_Exit(0);
    }
    return 2;
}
