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
    uint64_t id = 0;             // unique per decoded frame
};

class FrameCache {
public:
    static FrameCache& get();
    // Returns the decoded frame (waiting for an in-flight read-ahead), or null + error.
    std::shared_ptr<const Frame> fetch(const std::string& path, std::string& error);
    // Replaces the read-ahead queue with these paths (nearest first).
    void prefetch(const std::vector<std::string>& paths);
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
