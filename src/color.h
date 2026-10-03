// DNG colour science: temperature/tint <-> xy (DNG SDK / Robertson), ColorMatrix
// interpolation, camera -> output-primaries matrices, output encodings.
#pragma once
#include "dng.h"

namespace sfp {

struct Mat3 { double m[9]; };

enum class Primaries { Rec709, P3D65, Rec2020, DaVinciWideGamut, ACES_AP0, ACES_AP1, ArriWideGamut3, XYZ_D65, Count };
// Where the camera -> XYZ matrices come from: the file's ColorMatrix1/2, or the chart-fitted Sigma fp
// pair (fp_chart_matrix.h) scaled to the neutrals the file's own matrices give (same white balance).
enum class CameraMatrix { AsRecorded, ChartFitted, Count };
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
ColorSetup color_setup(const DngInfo& info, bool asShot, double temp, double tint, Primaries out,
                       CameraMatrix cam = CameraMatrix::AsRecorded);

// The ColorMatrix pair a decode uses (exposed for the self-test).
struct CmPair { Mat3 c1, c2; int illum1 = 0, illum2 = 0; bool hasCM2 = false; double analog[3] = {1, 1, 1}; };
CmPair recorded_matrices(const DngInfo& info);
CmPair chart_fitted_matrices(const CmPair& recorded);

}  // namespace sfp
