// CPU version of the development kernels (kernels_cpu.cpp).
#pragma once
#include <string>

#include "develop.h"

namespace sfp {

// Develops into the host image `out` (RGBA float, dp.rowBytes per row, which may be negative).
// source: not null = the host's own picture (dp.W x dp.H, RGBA float, sourceRowBytes per row,
// bottom row first), resampled instead of developing f.
bool develop_cpu(const Frame& f, int quality, const PrepParams& pp, float dz, const DevelopParams& dp, void* out,
                 std::string& error, DevelopTiming* timing, const void* source = nullptr, int sourceRowBytes = 0);

}  // namespace sfp
