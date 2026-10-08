// Sigma fp CinemaDNG reader: metadata + raw CFA (uncompressed 12-bit packed strip or
// LJ92 256x256 tiles as written by the R52 in-camera encoder or Sigma stills).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sfp {

// Lens shading map: the GainMap of the DNG's OpcodeList3, as the camera writes it from the
// lens's own data. Always the colour shading; also the vignette when the camera's Vignetting
// compensation is on (the green plane is then above 1 towards the corners).
struct GainMap {
    int rows = 0, cols = 0, planes = 0;   // grid points; 1 plane, or 3: R, G, B
    int areaW = 0, areaH = 0;             // the area the camera names (used for its shape only)
    std::vector<float> gain;              // rows x cols x planes, row by row
    bool valid() const { return rows >= 2 && cols >= 2 && (planes == 1 || planes == 3); }
    // Largest gain of the brightness (green) plane: above 1 = the map holds a vignette.
    float brightness() const {
        float m = 1;
        for (size_t i = planes == 3 ? 1 : 0; i < gain.size(); i += static_cast<size_t>(planes)) m = gain[i] > m ? gain[i] : m;
        return m;
    }
};

// Lens distortion profile: where a point of the corrected picture was recorded, as a radial
// factor sum of c[i] r^i. From the WarpRectilinear opcode of OpcodeList3 (terms of the green
// plane), or from a profile file (lens_profile.h).
struct LensWarp {
    bool valid = false;
    double c[7] = {1, 0, 0, 0, 0, 0, 0};
    double cx = 0.5, cy = 0.5;   // optical centre as a fraction of the picture
    double radius = 0;           // r = 1 at this many sensor pixels from the centre; 0 = the whole sensor's half diagonal
};

struct DngInfo {
    int width = 0, height = 0, bps = 0, compression = 0;
    int tileW = 0, tileH = 0;                 // 0 for strips
    std::vector<uint64_t> offsets, counts;    // strip or tiles
    int cfa[4] = {0, 1, 1, 2};                // colour at (y&1)*2+(x&1): 0 R, 1 G, 2 B
    float black[4] = {0, 0, 0, 0};            // per CFA position
    float white = 4095;
    int cropX = 0, cropY = 0, cropW = 0, cropH = 0;   // DefaultCrop (relative to ActiveArea)
    int activeTop = 0, activeLeft = 0;
    double cm1[9] = {}, cm2[9] = {};
    bool hasCM1 = false, hasCM2 = false;
    int illum1 = 0, illum2 = 0;
    double neutral[3] = {1, 1, 1};            // AsShotNeutral
    double baselineExposure = 0;
    double analog[3] = {1, 1, 1};
    int iso = 0;
    double fps = 0;
    std::string model;
    double focalMm = 0, fNumber = 0, focusM = 0;   // EXIF, 0 = unknown
    std::string lensModel;
    GainMap shading;                          // rows == 0: the file has none
    LensWarp warp;
};

// Parse metadata from a file already read into memory.
bool parse_dng(const uint8_t* data, size_t size, DngInfo& info, std::string& error);

// Decode the raw CFA into out (width*height uint16, top-down). Uses up to `threads`
// threads for LJ92 tiles. Returns false with an error message on malformed data.
bool decode_raw(const uint8_t* data, size_t size, const DngInfo& info, uint16_t* out,
                int threads, std::string& error);

// Applies a shading map to info's decoded raw samples in place (black level kept, clipped samples left
// clipped). The frame shows the middle fracW x fracH of what the map covers (1 = all of it).
// The samples are left at a finer scale: info's black and white levels are multiplied to match.
void apply_shading(DngInfo& info, const GainMap& g, uint16_t* raw, double fracW, double fracH);

bool read_file(const std::string& path, std::vector<uint8_t>& data, std::string& error);

}  // namespace sfp
