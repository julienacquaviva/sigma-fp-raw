// Decoded raw-frame cache with background read-ahead (shared by all instances).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dng.h"

namespace sfp {

struct Frame {
    std::string path;
    DngInfo info;
    std::vector<uint16_t> raw;   // width*height CFA samples
    double readMs = 0, decodeMs = 0;
    bool shaded = false;         // the lens shading map has been applied to raw
    // The share of the lens profile's area (the whole 3:2 sensor) that the frame shows, across
    // and down; 1 = all of it. See shading_window() in frame_cache.cpp.
    double profileW = 1, profileH = 1;
    uint64_t id = 0;             // unique per decoded frame
};

class FrameCache {
public:
    static FrameCache& get();
    // Returns the decoded frame (waiting for an in-flight read-ahead), or null + error.
    // shading: with the frame's lens shading map (vignette, colour shading) applied to the raw
    // samples; kept as a frame of its own next to the plain one.
    // vignetteFile: a profile file (lens_profile.h) whose vignette replaces the frame's own; the
    // frame's colour shading is kept. A file that cannot be used leaves the frame's own map.
    std::shared_ptr<const Frame> fetch(const std::string& path, std::string& error, bool shading = false, const std::string& vignetteFile = std::string());
    // Replaces the read-ahead queue with these paths (nearest first).
    void prefetch(const std::vector<std::string>& paths, bool shading = false, const std::string& vignetteFile = std::string());
    void set_budget(size_t bytes);
    struct Stats { uint64_t hits, misses, waits, prefetched, failed; size_t bytes, frames; };
    Stats stats() const;
    ~FrameCache();

private:
    FrameCache();
    struct Impl;
    Impl* d;
};

// Sequence helpers: "..._000123.DNG" <-> (prefix, number, digits, suffix).
bool split_sequence(const std::string& path, std::string& prefix, long long& number, int& digits, std::string& suffix);
std::string sequence_path(const std::string& prefix, long long number, int digits, const std::string& suffix);

}  // namespace sfp
