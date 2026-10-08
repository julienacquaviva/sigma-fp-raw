// Development on the Mac's GPU with Metal (develop_metal.mm; only in builds made on a Mac, which
// define SFP_METAL). Everywhere else these do nothing.
#pragma once
#include <string>

#include "develop.h"

namespace sfp {

#ifdef SFP_METAL
// True when the default GPU runs the kernels (compiled at the first call; SFP_METAL=0 switches it off).
bool metal_available();
// The GPU's name, or why Metal is not used.
std::string metal_status();
// Same arguments and result as develop_cpu (develop_cpu.h).
bool develop_metal(const Frame& f, int quality, const PrepParams& pp, float dz, const DevelopParams& dp, void* out,
                   std::string& error, DevelopTiming* timing, const void* source = nullptr, int sourceRowBytes = 0);
#else
inline bool metal_available() { return false; }
inline std::string metal_status() { return "not in this build"; }
inline bool develop_metal(const Frame&, int, const PrepParams&, float, const DevelopParams&, void*, std::string&, DevelopTiming*,
                          const void* = nullptr, int = 0) { return false; }
#endif

}  // namespace sfp
