// Development of a decoded raw frame into an RGBA float image: on the GPU with CUDA, else on the CPU.
#pragma once
#include <memory>
#include <string>

#include "color.h"
#include "cuda_api.h"
#include "frame_cache.h"
#include "kernel_params.h"

namespace sfp {

enum class WhiteBalance { AsShot, Daylight, Cloudy, Shade, Tungsten, Fluorescent, Flash, Custom, Count };
enum class Fit { Fit, Fill, Stretch, Native, FitWidth, FitHeight, Count };

// The Transform controls, as in DaVinci Resolve's Transform panel. Lengths in timeline pixels,
// angles in degrees; position and anchor are measured from the frame centre with y up.
struct Transform {
    double zoomX = 1, zoomY = 1;
    double posX = 0, posY = 0;
    double rotation = 0;              // positive turns the picture anticlockwise
    double anchorX = 0, anchorY = 0;  // the point zoom, rotation, pitch and yaw turn about
    double pitch = 0, yaw = 0;        // the picture as a card turned about its horizontal / vertical axis, seen in perspective
    bool flipH = false, flipV = false;
    bool neutral() const {
        return zoomX == 1 && zoomY == 1 && posX == 0 && posY == 0 && rotation == 0 && pitch == 0 && yaw == 0 && !flipH && !flipV;
    }
};

// Camera RAW controls (DaVinci Resolve's CinemaDNG panel layout and ranges) plus
// Sigma fp specific options.
struct RawSettings {
    int decodeQuality = 0;                 // 0 full (RCD), 1 half
    WhiteBalance whiteBalance = WhiteBalance::AsShot;
    double colorTemp = 5600, tint = 0;     // used for presets/custom
    Primaries colorSpace = Primaries::Rec709;
    Gamma gamma = Gamma::Rec709;           // linear toe: no black sparkle (pure 2.4 amplifies noise at 0)
    double exposure = 0;                   // stops, -5..5
    double sharpness = 0;                  // 0..100
    double highlights = 0, shadows = 0;    // -100..100
    double colorBoost = 0;                 // 0..100
    double saturation = 50;                // 0..100 (50 = neutral)
    double midtones = 0;                   // -100..100
    double lift = 0, gain = 0;             // -100..100
    double contrast = 50;                  // 0..100 (50 = neutral)
    bool highlightRecovery = false, gamutMapping = false, preToneCurve = false, softClip = false;
    double deZigzag = 80;                  // Edge Anti-aliasing 0..100 (3K only; 80 recommended)
    bool hotPixels = true;                 // remove isolated hot pixels
    double rowPhase = -0.125;              // 3K only; same-colour plane pitch (-0.125 recommended)
    // Whether the 3024x2010 frame comes from the 2x2 binned readout (sensor mode M98), the only one
    // with the row-pair offset: 1 yes, 0 no (a 1:1 sensor window of the same size), -1 unknown
    // (decided by the frame size, as before 1.4.2). See frame_binning() in gyro.h.
    int binned = -1;
    Fit fit = Fit::Fit;
    Transform xf;                          // applied in the same resampling as the fit and the stabilisation
    int resampler = kResampleLanczos3;     // the kernel of that resampling (kernel_params.h)
    bool lensDistortion = false;           // correct the lens distortion, in the same resampling
    LensWarp lensProfile;                  // with this profile (lens_distortion() in lens_profile.h); not valid = none
    // Develop RAW off only: the vignette to remove from the host's picture (a brightness map over
    // the whole sensor, lens_vignette() in lens_profile.h; not valid = none) and the encoding of
    // that picture (ids of pt_decode in kernels.cu).
    GainMap sourceVignette;
    int sourceGamma = 4;
    double renderScaleX = 1, renderScaleY = 1;
    StabParams stab{};                     // gyro stabilisation warp of this frame (gyro.h); on == 0: none
};

// Row phase used for a frame (auto = measured M98 3K readout value).
double effective_row_phase(const RawSettings& s, const DngInfo& info);
double effective_dezigzag(const RawSettings& s, const DngInfo& info);
// White balance as temperature/tint for a preset (AsShot/Custom return false).
bool preset_temp_tint(WhiteBalance wb, double& temp, double& tint);

struct Target {
    CUdeviceptr device = 0;   // device image (host CUDA render), or 0
    void* host = nullptr;     // host image (CPU render)
    int width = 0, height = 0, rowBytes = 0;
    int boundsX = 0, boundsY = 0;
    int rodW = 0, rodH = 0;   // region of definition the frame is fitted into
    // The timeline frame in the region's pixels, centred on it, when it has another shape than
    // the region (0 = it is the region). Fit Width / Fit Height fit this frame.
    double compW = 0, compH = 0;
};

// The host's own picture of the clip (its Source image), for developing switched off: the
// plug-in then only frames, stabilises and straightens that picture. The clip's frame is taken
// to be the whole of that picture.
struct SourceImage {
    CUdeviceptr device = 0;      // device image (host CUDA render), or 0
    const void* host = nullptr;  // host image
    int width = 0, height = 0, rowBytes = 0;   // RGBA float, bottom row first
    int offX = 0, offTop = 0;    // the image's left / top edge within its region of definition
    int rodW = 0, rodH = 0;
};

// Fit and Transform for an output frame: fills the geometry part of p (scale, offsets, xform).
// Also used by the command-line tool and the tests.
void frame_geometry(int cropW, int cropH, const RawSettings& s, const Target& t, DevelopParams& p);
// The resampling kernel and its widening for the frame (after frame_geometry, with crop and stab set).
void resample_scale(const RawSettings& s, DevelopParams& dp);

struct DevelopTiming { double uploadMs = 0, gpuMs = 0; bool demosaicReused = false; };

class Developer {
public:
    // stream may be null (synchronous). When called with host CUDA images the host's
    // context must be current; otherwise the device's primary context is used, or the CPU
    // when there is no CUDA driver.
    // source: not null = do not develop the raw frame; resample this picture instead.
    bool develop(const Frame& f, const RawSettings& s, CUstream stream, const Target& t,
                 std::string& error, DevelopTiming* timing = nullptr, const SourceImage* source = nullptr);
    static Developer& get();
    // "CUDA (GPU)", "Metal (GPU)" or "CPU": what develops host images on this machine.
    static const char* backend();
    // Tests: true = develop on the processor whatever the machine has (as SFP_CPU=1 does); false = back to normal.
    static void force_cpu(bool on);

private:
    struct Impl;
    std::shared_ptr<Impl> d;
    Developer();
};

}  // namespace sfp
