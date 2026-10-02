// Sigma fp CinemaDNG reader: metadata + raw CFA (uncompressed 12-bit packed strip or
// LJ92 256x256 tiles as written by the R52 in-camera encoder or Sigma stills).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace sfp {

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
};

// Parse metadata from a file already read into memory.
bool parse_dng(const uint8_t* data, size_t size, DngInfo& info, std::string& error);

// Decode the raw CFA into out (width*height uint16, top-down). Uses up to `threads`
// threads for LJ92 tiles. Returns false with an error message on malformed data.
bool decode_raw(const uint8_t* data, size_t size, const DngInfo& info, uint16_t* out,
                int threads, std::string& error);

bool read_file(const std::string& path, std::vector<uint8_t>& data, std::string& error);

}  // namespace sfp
