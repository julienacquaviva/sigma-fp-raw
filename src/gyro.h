// Gyro stabilisation from the camera's gyro data: either "FPG2" blocks inside every DNG frame
// (gyro_inframe.cpp) or the older FPGY sidecar (<clip>.FPG next to the DNGs). Both give the
// raw gyro samples of the clip and, per frame, the sample count at the frame's mark. The
// angular rate is integrated into a camera orientation path; the path
// sampled at each frame's readout middle is low-pass filtered, and the difference between
// the smoothed and the actual orientation is applied as an image warp (pinhole model, no
// lens distortion), per sensor row for the rolling-shutter correction.
#pragma once
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "kernel_params.h"

namespace sfp {

struct Quat { double w = 1, x = 0, y = 0, z = 0; };

struct GyroHeader {
    int version = 0;
    uint32_t frameCount = 0, sampleCount = 0;
    double rateHz = 0;            // samples per second (measured by the camera for this clip)
    double lsbPerDps = 131;       // counts per degree/second
    int rasterW = 0, rasterH = 0; // DNG raster
    int activeX = 0, activeY = 0, activeW = 0, activeH = 0;
    double fps = 0;
    double readoutS = 0;          // first to last raster row
    double exposureS = 0;
    double focalMm = 0;           // 0 = unknown (manual lens)
    double markDelayS = 0;        // readout middle -> frame mark
    int axis[3] = {2, -1, 3};     // pan, tilt, roll: +-(gyro axis + 1)
    unsigned flags = 0;           // bit0: capture stopped before the end of the clip
    uint32_t preroll = 0;
    double pitchNm = 0;           // pixel pitch at raster scale, 0 = unknown
    uint32_t resolution = 0, dcCrop = 0, bitDepth = 0, sensorMode = 0;
};

// One frame's in-frame gyro block ("FPG2", FPGYRO_INFRAME.md).
struct GyroBlock {
    bool present = false;
    bool fileExists = false;      // set by the clip scan: a DNG with this number is on disk
    uint32_t frame = 0;           // number of the frame in the take
    uint32_t clockUs = 0;         // camera clock at the mark, wraps at 2^32
    uint32_t take = 0;
    unsigned flags = 0;           // bit0 first frame (pre-roll), bit1 samples lost before this slice
    unsigned lost = 0, ringIndex = 0;
    uint32_t readoutUs = 0;
    int32_t markDelayUs = 0;
    unsigned sensorMode = 0, resolution = 0, dcCrop = 0, bitDepth = 0, lsb = 0;
    int axis[3] = {0, 0, 0};
    std::vector<int16_t> samples; // n x (X, Y, Z)
};
const int kFpg2Region = 0xC00;    // the block sits in the last 3072 bytes of the MakerNote
const int kFpg2MaxSamples = 500;

// Parses the block region of one frame. False (with a reason) when there is no valid block.
bool parse_fpg2(const uint8_t* region, size_t size, GyroBlock& out, std::string* why = nullptr);
// Reads the block of one DNG file: fixed offset first, then through IFD0 -> EXIF -> MakerNote.
bool read_fpg2(const std::string& dngPath, GyroBlock& out, std::string* why = nullptr);

// What the DNG itself tells about the clip (not repeated in the blocks).
struct ClipGeometry {
    int rasterW = 0, rasterH = 0, activeX = 0, activeY = 0, activeW = 0, activeH = 0;
    double fps = 0, focalMm = 0, exposureS = 0;
};

// User controls (the OFX "Stabilisation" group).
struct StabSettings {
    bool enable = true;
    double smoothness = 0.5;      // Gaussian time constant of the path low-pass, seconds (0 = follow the camera)
    double rollingShutter = 1.0;  // 0..1 share of the readout time that is corrected
    double syncMs = 0;            // added to the file's mark delay
    double focalMm = 0;           // 0 = from the file
    double readoutMs = 0;         // readout time of the raster rows; 0 = from the file
    bool autoZoom = true;         // zoom in so that no frame of the clip shows a border
    double maxZoom = 1.3;         // limit of the automatic zoom; frames that need more are stabilised less
    double zoom = 1.0;            // manual zoom, multiplies the automatic one
    long long clipFrames = 0;     // DNG frames of the clip (0 = unknown); filled in by gyro_for_frame
    // The part of the clip that is used (a trimmed or bladed timeline clip): 0-based frames,
    // inclusive; rangeLast < 0 = the whole clip. Smoothing, zoom and the zoom limit look at
    // this part only, as if it were the clip. Frames outside it get its zoom and as much
    // correction as fits.
    long long rangeFirst = 0, rangeLast = -1;
    std::string rangeNote;        // where the range comes from, for the status line
    bool rawMarks = false;        // measurements only: take each frame's time from its own mark (as before v1.3.1)
    int zoomMode = 0;             // 0 fixed: one zoom for the range; 1 dynamic: follows the local need
    double zoomSmooth = 4.0;      // dynamic zoom: seconds over which it looks ahead and eases
};

class GyroClip {
public:
    // Parses a whole FPGY file; null + error when it is not a valid version 1 file.
    static std::shared_ptr<GyroClip> parse(const uint8_t* data, size_t size, std::string& error);

    // Builds the track from the blocks of a clip; blocks[i] belongs to DNG number i + 1.
    static std::shared_ptr<GyroClip> from_blocks(const std::vector<GyroBlock>& blocks, const ClipGeometry& geo, std::string& error);
    // The stored form of a track (what parse and from_blocks end in, and what the cache file holds).
    static std::shared_ptr<GyroClip> assemble(const GyroHeader& h, std::vector<uint32_t> table, std::vector<uint8_t> valid,
                                              const int16_t* samples, std::string& error);
    const std::vector<uint32_t>& table() const { return table_; }
    const std::vector<uint8_t>& valid() const { return valid_; }
    const std::vector<int16_t>& raw() const { return raw_; }   // in-frame tracks only, until the cache file is written
    void drop_raw() { std::vector<int16_t>().swap(raw_); }
    bool has_data(long long i) const { return i >= 0 && i < frames() && valid_[static_cast<size_t>(i)]; }

    GyroHeader h;
    std::string path, name;
    bool inFrame = false;         // built from FPG2 blocks
    int blocks = 0;               // frames with a usable block
    int noData = 0;               // frames without gyro data (not stabilised)
    long long lostSamples = 0;    // samples the camera reported lost (filled as gaps)

    int frames() const { return static_cast<int>(table_.size()); }
    // Time (seconds on the sample clock) of the middle of frame i's readout. The sensor exposes
    // frames on a regular cadence; the camera's mark of a frame can be late (never early) when
    // its recorder task is held up, by tens of milliseconds at worst. So the time is taken from
    // the cadence fitted to the marks (regular_marks), not from the frame's own mark.
    double frame_time(long long i, double syncMs, bool rawMarks = false) const;
    // Mark of frame i as recorded, and on the fitted cadence (seconds on the sample clock).
    double mark_recorded(long long i) const { return table_[static_cast<size_t>(std::clamp<long long>(i, 0, frames() - 1))] / h.rateHz; }
    double mark_regular(long long i) const { return mark_[static_cast<size_t>(std::clamp<long long>(i, 0, frames() - 1))]; }
    // Time from the middle of frame i's readout to its (undisturbed) mark. The mark is set
    // when the recorder gets the frame, about at the end of the readout, and the picture
    // belongs to the middle of the exposure: readout/2 + exposure/2 + a small constant. The
    // exposure time is the DNG's EXIF ExposureTime. Without it (or without a readout time) the
    // value written with the gyro data is used (the camera's provisional readout/2 + 5 ms).
    double delay_s(long long i) const;
    double exposure_s(long long i) const;   // 0 = unknown
    bool delay_from_exposure() const { return h.readoutS > 0 && exposure_s(0) > 0; }
    std::vector<float> exposureByFrame;     // only when the exposure time changes inside the clip
    int droppedFrames = 0;        // frames the camera dropped inside the take (a gap in the cadence)
    double typicalLateMs = 0;     // how late an undisturbed mark is against the earliest ones (median)
    int lateMarks = 0;            // marks more than 5 ms late
    // Camera orientation (camera -> world) at time t, from the integrated rate.
    Quat orientation(double t) const;
    // Orientation of the virtual (stabilised) camera at frame i: the smoothed path, pulled
    // towards the real camera where the automatic zoom would pass its limit.
    Quat smoothed(long long i, const StabSettings& s) const;
    // Frames whose correction was reduced to stay within the zoom limit.
    int limited_frames(const StabSettings& s) const;
    // Readout time (first to last raster row) the rolling-shutter correction uses, seconds.
    double readout_s(const StabSettings& s) const { return s.readoutMs > 0 ? s.readoutMs * 1e-3 : h.readoutS; }
    // Focal length in raster pixels; 0 with a reason when it is unknown.
    double focal_px(const StabSettings& s, std::string* why = nullptr) const;
    // Zoom that hides the borders on every frame of the used range (1 = none needed); with
    // the dynamic zoom its largest value.
    double auto_zoom(const StabSettings& s) const;
    // Smallest, mean and largest automatic zoom over the used range.
    void zoom_stats(const StabSettings& s, double& lo, double& mean, double& hi) const;
    // Whether the smoothing runs on past the start / the end of the used range (see ensure_zoom).
    void range_ends(const StabSettings& s, bool& pastStart, bool& pastEnd) const;
    // The used range as 0-based frames, clamped to the track.
    void used_range(const StabSettings& s, int& first, int& last) const;
    // True when the gyro data stops before the end of the clip (fewer gyro frames than DNGs,
    // or, with an unknown DNG count, the file's "stopped early" flag). The correction, the
    // rolling-shutter correction and the automatic zoom then fade to none over the last
    // second of gyro data, so that the picture does not jump where the data ends.
    bool ends_early(const StabSettings& s) const;
    // Automatic zoom of frame i (the clip's zoom, less inside the fade-out).
    double frame_zoom(long long i, const StabSettings& s) const;
    // Warp of frame i (0-based). Frames outside the file are not warped (only the manual zoom).
    // Returns false (out.on = 0) with a reason when stabilisation cannot run.
    bool warp(long long i, const StabSettings& s, StabParams& out, std::string& why) const;
    // One line for the status field.
    std::string status(const StabSettings& s) const;

private:
    std::vector<uint32_t> table_;   // samples written up to each frame's mark
    std::vector<uint8_t> valid_;    // frames with gyro data
    std::vector<double> mark_;      // each frame's mark on the regular cadence, seconds
    std::vector<int16_t> raw_;
    std::vector<Quat> q_;           // orientation at each sample time
    struct Path {
        bool valid = false, rawMarks = false;
        double smoothness = 0, syncMs = 0;
        int first = 0, last = 0;
        std::vector<double> time;
        std::vector<Quat> actual, smooth;
    };
    // Smoothed path with the range's ends treated as clip ends (false) or with the frames
    // beyond that end taken into the smoothing (true).
    void smooth_path(bool pastStart, bool pastEnd, std::vector<Quat>& out) const;   // m_ held, path_.actual valid
    struct Zoom {
        bool valid = false, autoZoom = false, fade = false;
        double rs = 0, focal = 0, maxZoom = 0, value = 1, low = 1, mean = 1, smooth = 0, readout = 0;
        int limited = 0, mode = 0;
        bool pastStart = false, pastEnd = false;   // how the range's ends were smoothed (the cheaper way)
        std::vector<Quat> virt;       // virtual camera per frame
        std::vector<float> zoom, rs_; // automatic zoom and rolling-shutter amount per frame
    };
    mutable std::mutex m_;
    mutable Path path_;
    mutable Zoom zoom_;
    void ensure_path(const StabSettings& s) const;                 // m_ held
    void ensure_zoom(const StabSettings& s, double focal) const;   // m_ held
    void rotations(long long i, const StabSettings& s, double rs, const Quat& virt, float* rot) const;
};

// The clip's sidecar: an .FPG in the DNG's folder whose name starts the DNG's name
// (A001_092.FPG for A001_092_20261001_000001.DNG) or equals the folder's name, or the only
// .FPG there. Empty + reason when there is none.
std::string find_fpg(const std::string& dngPath, std::string& why);

// In-frame gyro of the clip a DNG belongs to. The first call starts a background scan of the
// sequence and returns at once; `loading` with a percentage until it is done. Then `clip` is
// the track, or null when the frames carry no gyro blocks. Results are kept in memory and in
// a cache file under %LOCALAPPDATA%\SigmaFpRaw\gyro (SFP_GYRO_CACHE_DIR overrides, "off" disables).
struct GyroLoad {
    bool loading = false;
    int percent = 0;
    long long frames = 0;         // DNG files of the clip
    std::shared_ptr<const GyroClip> clip;
    bool fromCache = false;
    double scanMs = 0;            // time the scan (or the cache read) took
    std::string note;             // blocks were found but could not be used: why
};
GyroLoad inframe_gyro(const std::string& dngPath);
GyroLoad inframe_gyro_wait(const std::string& dngPath);   // blocks until the scan is done (tools, tests)
// Reads the block region of every frame of a clip, as the scan does, without building or
// caching anything (timing tool). Returns the milliseconds taken.
double scan_blocks(const std::string& dngPath, long long& files, long long& found, int threads = 0, bool fixedOffsetOnly = false,
                   bool unbuffered = false);
void gyro_shutdown();                                      // stops background scans (before the library unloads)

// Loads (and caches) a sidecar file. dngPath (a frame of the clip) supplies the exposure time
// when the file does not carry it.
std::shared_ptr<const GyroClip> load_gyro(const std::string& fpgPath, std::string& error, const std::string& dngPath = std::string());
// EXIF ExposureTime of a DNG in seconds (0 = unknown).
double dng_exposure(const std::string& dngPath);

// Everything a host needs for one frame. Gyro source: gyroFile when not empty, else the
// blocks inside the clip's DNGs, else the .FPG sidecar next to them. Never blocks: while the
// in-frame data is being gathered the frame is not warped and the status says "Loading gyro".
// Checks the track against the DNG raster and fills the warp of 0-based frame `frame`.
// Returns the clip (null when there is none); `status` is the text of the status field and
// warp.on tells whether the frame is warped. `wait` (tools, tests) waits for the scan.
std::shared_ptr<const GyroClip> gyro_for_frame(const std::string& dngPath, const std::string& gyroFile, long long frame,
                                               int dngW, int dngH, const StabSettings& s, StabParams& warp, std::string& status,
                                               bool wait = false);

}  // namespace sfp
