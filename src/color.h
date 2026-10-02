// DNG colour science: temperature/tint <-> xy (DNG SDK / Robertson), ColorMatrix
// interpolation, camera -> output-primaries matrices, output encodings.
#pragma once
#include "dng.h"

namespace sfp {

struct Mat3 { double m[9]; };

enum class Primaries { Rec709, P3D65, Rec2020, DaVinciWideGamut, ACES_AP0, ACES_AP1, XYZ_D65, Count };
enum class Gamma { Linear, Gamma22, Gamma24, Rec709, sRGB, DaVinciIntermediate, ACEScct, Count };

void xy_to_temp_tint(double x, double y, double& temp, double& tint);
void temp_tint_to_xy(double temp, double tint, double& x, double& y);

struct ColorSetup {
    float wb[3];          // per-channel multipliers applied to linear raw (green = 1)
    float matrix[9];      // white-balanced camera RGB -> linear output primaries (row-major)
    double temp, tint;    // white point of the balance (as shot or custom)
    float lum[3];         // luminance (Y) weights of the output primaries
};

// asShot: use AsShotNeutral; otherwise use temp/tint.
ColorSetup color_setup(const DngInfo& info, bool asShot, double temp, double tint, Primaries out);

}  // namespace sfp
