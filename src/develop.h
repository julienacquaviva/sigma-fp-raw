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
enum class Fit { Fit, Fill, Stretch, Native, Count };

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
    Fit fit = Fit::Fit;
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
};

struct DevelopTiming { double uploadMs = 0, gpuMs = 0; bool demosaicReused = false; };

class Developer {
public:
    // stream may be null (synchronous). When called with host CUDA images the host's
    // context must be current; otherwise the device's primary context is used, or the CPU
    // when there is no CUDA driver.
    bool develop(const Frame& f, const RawSettings& s, CUstream stream, const Target& t,
                 std::string& error, DevelopTiming* timing = nullptr);
    static Developer& get();
    // "CUDA (GPU)" or "CPU": what develops host images on this machine.
    static const char* backend();

private:
    struct Impl;
    std::shared_ptr<Impl> d;
    Developer();
};

}  // namespace sfp
