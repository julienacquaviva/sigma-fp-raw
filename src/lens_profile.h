// Lens correction profiles read from a file: a DNG of the same lens, an Adobe lens profile
// (.lcp) or a Lensfun database file (.xml). See lens_profile.cpp.
#pragma once
#include <string>

#include "dng.h"

namespace sfp {

// The vignette of the profile as a brightness map over the whole sensor, for the clip's focal
// length, aperture and focus distance (the nearest calibration point). False + the reason in
// note when the file cannot be used; on success note says what was taken ("Adobe profile, 28 mm f/4.0").
bool profile_vignette(const std::string& file, const DngInfo& clip, GainMap& out, std::string& note);
// The same for the distortion.
bool profile_distortion(const std::string& file, const DngInfo& clip, LensWarp& out, std::string& note);

// When the clip carries no profile of its own: the Adobe lens profile installed on this machine
// (Camera Raw / Lightroom / Photoshop) whose name fits the clip's lens and which holds a vignette
// (or a distortion) model at the clip's focal length; or empty. Looked up once per lens.
std::string adobe_profile_for(const DngInfo& clip, bool vignette);

// What the two corrections use for a clip, in this order: the file the user picked; else what
// the frames carry; else the installed Adobe profile of the lens. False = nothing to correct
// with (out is then not valid). note: where it came from, or why a file was not used.
bool lens_vignette(const std::string& userFile, const DngInfo& clip, GainMap& out, bool& ownColour, std::string& note);
bool lens_distortion(const std::string& userFile, const DngInfo& clip, LensWarp& out, std::string& note);

}  // namespace sfp
