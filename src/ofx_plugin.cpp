// Sigma fp RAW: OpenFX effect for DaVinci Resolve. It re-reads the clip's original
// CinemaDNG frames (uncompressed or LJ92-tiled), develops them on the GPU inside
// Resolve's CUDA context, and replaces its input with the result. Camera RAW controls
// follow Resolve's CinemaDNG panel; Sigma fp controls add the 3K row-phase correction
// and edge anti-aliasing.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxMessage.h"
#include "ofxParam.h"
#include "ofxProperty.h"

#include "develop.h"
#include "gyro.h"
#include "lens_profile.h"
#include "platform.h"

// CUDA rendering (ofxGPURender.h) and DaVinci Resolve's additions to OpenFX (its ofxImageEffectExt.h):
// property names only.
#ifndef kOfxImageEffectPropCudaRenderSupported
#define kOfxImageEffectPropCudaRenderSupported "OfxImageEffectPropCudaRenderSupported"
#endif
#ifndef kOfxImageEffectPropCudaEnabled
#define kOfxImageEffectPropCudaEnabled "OfxImageEffectPropCudaEnabled"
#endif
#ifndef kOfxImageEffectPropCudaStreamSupported
#define kOfxImageEffectPropCudaStreamSupported "OfxImageEffectPropCudaStreamSupported"
#endif
#ifndef kOfxImageEffectPropCudaStream
#define kOfxImageEffectPropCudaStream "OfxImageEffectPropCudaStream"
#endif
#ifndef kOfxImageEffectPropNoSpatialAwareness
#define kOfxImageEffectPropNoSpatialAwareness "OfxImageEffectPropNoSpatialAwareness"
#endif
#ifndef kOfxImageEffectPropSrcFilePath
#define kOfxImageEffectPropSrcFilePath "OfxImageEffectPropSrcFilePath"
#endif
#ifndef kOfxImageEffectPropSrcFrame
#define kOfxImageEffectPropSrcFrame "OfxImageEffectPropSrcFrame"
#endif

#ifndef SFP_VERSION
#define SFP_VERSION "dev"
#endif

using namespace sfp;

namespace {

OfxHost* gHost = nullptr;
const OfxPropertySuiteV1* gProp = nullptr;
const OfxImageEffectSuiteV1* gEffect = nullptr;
const OfxParameterSuiteV1* gParam = nullptr;
const OfxMessageSuiteV2* gMsg2 = nullptr;
const OfxMessageSuiteV1* gMsg1 = nullptr;


// ---- parameter names ----
const char* WB_OPTIONS[] = {"As Shot", "Daylight", "Cloudy", "Shade", "Tungsten", "Fluorescent", "Flash", "Custom"};
const char* CS_OPTIONS[] = {"Rec.709", "P3 D65", "Rec.2020", "DaVinci Wide Gamut", "ACES AP0", "ACES AP1"};
const char* GAMMA_OPTIONS[] = {"Linear", "Gamma 2.2", "Gamma 2.4", "Rec.709", "sRGB", "DaVinci Intermediate", "ACEScct"};

struct Instance {
    OfxImageEffectHandle handle{};
    OfxImageClipHandle output{}, source{};
    std::map<std::string, OfxParamHandle> params;
    std::mutex m;
    long long lastFrame = -1;
    int direction = 1;
    std::string metaPath;
    bool haveMeta = false;
    DngInfo meta;
    bool updating = false;
    std::string stabStatus;   // last text written to the status field
    // What the host said about the used part of the clip at the last render (Range: Automatic).
    std::mutex rangeM;
    bool hostKnown = false, hostRetimed = false, hostHasRange = false;
    double hostOffset = 0;            // source frame (0-based) minus effect time
    double hostFirst = 0, hostLast = 0;   // frame range of the source clip in effect time
    std::string hostSource;           // clip the above belongs to
    // The frame the host asked for at the last render and its project (timeline) frame, in
    // render-scaled pixels; 0 = not known yet. For the Clip Info block.
    int lastRodW = 0, lastRodH = 0;
    double lastProjW = 0, lastProjH = 0;
    // EXIF of the first frame (0 = unknown).
    double exposureS = 0, exifFocalMm = 0;
    int windowW = 0, windowH = 0;     // sensor window, from the first frame's gyro block (0 = none)
    OfxPropertySetHandle zoomSmoothProps = nullptr;   // to grey Smoothness (zoom) out while Zoom Mode is Fixed
};

OfxPropertySetHandle effect_props(OfxImageEffectHandle h) {
    OfxPropertySetHandle p = nullptr;
    gEffect->getPropertySet(h, &p);
    return p;
}
Instance* instance(OfxImageEffectHandle h) {
    void* v = nullptr;
    gProp->propGetPointer(effect_props(h), kOfxPropInstanceData, 0, &v);
    return static_cast<Instance*>(v);
}
void set_s(OfxPropertySetHandle p, const char* n, const char* v, int i = 0) { gProp->propSetString(p, n, i, v); }
void set_i(OfxPropertySetHandle p, const char* n, int v, int i = 0) { gProp->propSetInt(p, n, i, v); }
void set_d(OfxPropertySetHandle p, const char* n, double v, int i = 0) { gProp->propSetDouble(p, n, i, v); }
std::string get_s(OfxPropertySetHandle p, const char* n) {
    char* v = nullptr;
    return gProp->propGetString(p, n, 0, &v) == kOfxStatOK && v ? v : "";
}

void post_error(OfxImageEffectHandle h, const std::string& text) {
    if (gMsg2) gMsg2->setPersistentMessage(h, kOfxMessageError, "sfp_error", "%s", text.c_str());
    else if (gMsg1) gMsg1->message(h, kOfxMessageError, "sfp_error", "%s", text.c_str());
}
void clear_error(OfxImageEffectHandle h) {
    if (gMsg2) gMsg2->clearPersistentMessage(h);
}

// ---- describe ----
struct Definer {
    OfxParamSetHandle ps;
    OfxPropertySetHandle page;
    int child = 0;
    OfxPropertySetHandle last = nullptr;
    // Hides the parameter defined last: it stays in the project (old projects load and render as
    // before, with whatever value they stored) but has no control in the panel.
    void hide() { set_i(last, kOfxParamPropSecret, 1); }
    OfxPropertySetHandle define(const char* type, const char* name, const char* label, const char* hint, const char* parent) {
        OfxPropertySetHandle p = nullptr;
        gParam->paramDefine(ps, type, name, &p);
        last = p;
        set_s(p, kOfxPropLabel, label);
        set_s(p, kOfxParamPropHint, hint);
        if (parent) set_s(p, kOfxParamPropParent, parent);
        set_s(page, kOfxParamPropPageChild, name, child++);
        return p;
    }
    void group(const char* name, const char* label, bool open, const char* hint) {
        auto p = define(kOfxParamTypeGroup, name, label, hint, nullptr);
        set_i(p, kOfxParamPropGroupOpen, open ? 1 : 0);
    }
    void number(const char* name, const char* label, double def, double lo, double hi, double dlo, double dhi,
                const char* parent, const char* hint, int digits = 2) {
        auto p = define(kOfxParamTypeDouble, name, label, hint, parent);
        set_d(p, kOfxParamPropDefault, def);
        set_d(p, kOfxParamPropMin, lo); set_d(p, kOfxParamPropMax, hi);
        set_d(p, kOfxParamPropDisplayMin, dlo); set_d(p, kOfxParamPropDisplayMax, dhi);
        set_i(p, kOfxParamPropDigits, digits);
        set_s(p, kOfxParamPropDoubleType, kOfxParamDoubleTypePlain);
    }
    void choice(const char* name, const char* label, const char* const* opts, int n, int def, const char* parent, const char* hint) {
        auto p = define(kOfxParamTypeChoice, name, label, hint, parent);
        for (int i = 0; i < n; ++i) set_s(p, kOfxParamPropChoiceOption, opts[i], i);
        set_i(p, kOfxParamPropDefault, def);
    }
    void toggle(const char* name, const char* label, bool def, const char* parent, const char* hint) {
        auto p = define(kOfxParamTypeBoolean, name, label, hint, parent);
        set_i(p, kOfxParamPropDefault, def ? 1 : 0);
    }
    OfxPropertySetHandle label(const char* name, const char* label, const char* parent, const char* hint) {
        auto p = define(kOfxParamTypeString, name, label, hint, parent);
        set_s(p, kOfxParamPropStringMode, kOfxParamStringIsLabel);
        set_s(p, kOfxParamPropDefault, "");
        set_i(p, kOfxParamPropPersistant, 0);
        set_i(p, kOfxParamPropEvaluateOnChange, 0);
        set_i(p, kOfxParamPropAnimates, 0);
        return p;
    }
};

OfxStatus describe(OfxImageEffectHandle h) {
    auto p = effect_props(h);
    set_s(p, kOfxPropLabel, "Sigma fp RAW");
    set_s(p, kOfxImageEffectPluginPropGrouping, "Sigma fp");
    set_s(p, kOfxPropPluginDescription,
          "GPU CinemaDNG developer for Sigma fp clips (uncompressed and LJ92 compressed frames), with Camera RAW "
          "controls, 3K row-phase correction and edge anti-aliasing. Apply as the first node of the original DNG clip.");
    set_s(p, kOfxImageEffectPropSupportedContexts, kOfxImageEffectContextFilter);
    set_s(p, kOfxImageEffectPropSupportedPixelDepths, kOfxBitDepthFloat);
    set_s(p, kOfxImageEffectPluginRenderThreadSafety, kOfxImageEffectRenderInstanceSafe);
    set_i(p, kOfxImageEffectPluginPropHostFrameThreading, 0);
    set_i(p, kOfxImageEffectPropSupportsTiles, 0);
    set_i(p, kOfxImageEffectPropSupportsMultiResolution, 1);
    set_i(p, kOfxImageEffectPropTemporalClipAccess, 0);
    set_s(p, kOfxImageEffectPropNoSpatialAwareness, "false");
    set_s(p, kOfxImageEffectPropCudaRenderSupported, "true");
    set_s(p, kOfxImageEffectPropCudaStreamSupported, "true");
    return kOfxStatOK;
}

OfxStatus describe_in_context(OfxImageEffectHandle h) {
    for (const char* name : {kOfxImageEffectSimpleSourceClipName, kOfxImageEffectOutputClipName}) {
        OfxPropertySetHandle c = nullptr;
        gEffect->clipDefine(h, name, &c);
        set_s(c, kOfxImageEffectPropSupportedComponents, kOfxImageComponentRGBA);
        set_i(c, kOfxImageEffectPropSupportsTiles, 0);
    }
    OfxParamSetHandle ps = nullptr;
    gEffect->getParamSet(h, &ps);
    OfxPropertySetHandle page = nullptr;
    gParam->paramDefine(ps, kOfxParamTypePage, "Controls", &page);
    Definer d{ps, page};
    d.group("clipInfo", "Clip Info (v" SFP_VERSION ")", true, "What the clip was shot with and how it is framed. Refreshes when the node is created and when a control is changed.");
    d.label("info", "Clip", "clipInfo", "First DNG of the clip.");
    d.label("infoExposure", "Exposure", "clipInfo", "Shutter speed, shutter angle, ISO and frame rate.");
    d.label("infoFormat", "Format", "clipInfo", "Aspect ratio, bit depth, the sensor window that was read and the recorded (scaled) resolution.");
    d.label("infoLens", "Lens", "clipInfo",
            "Focal length and crop factor against full frame at the current frame: sensor window, fit, Transform zoom and the stabiliser's zoom "
            "(rolling-shutter correction included). Refreshes when a control is changed, not during playback.");
    // Framing: the fit, then a transform like DaVinci Resolve's Transform panel. Both are part of the
    // one resampling that also stabilises, from the full-resolution developed picture.
    d.group("xform", "Transform", false, "Zoom, position, rotation, anchor, pitch, yaw and flip, applied in the same single resampling as the "
                                         "fit and the stabilisation (no second resampling). Same controls and units as Resolve's Transform.");
    const char* fits[] = {"Scale to Fit", "Fit Width", "Fit Height"};
    d.choice("fitMode", "Fit", fits, 3, 0, "xform",
             "How the picture maps onto the timeline frame before the Transform. Scale to Fit: the whole picture, as Resolve's input "
             "scaling does. Fit Width / Fit Height: the width / the height fills the timeline frame. A Transform zoom of 1 is this exact fit.");
    d.number("xfZoomX", "Zoom X", 1, 0, 100, 0, 4, "xform", "Horizontal zoom about the anchor point; 1 = the fit.", 3);
    d.number("xfZoomY", "Zoom Y", 1, 0, 100, 0, 4, "xform", "Vertical zoom about the anchor point; follows Zoom X while Link Zoom is on.", 3);
    d.toggle("xfZoomLink", "Link Zoom", true, "xform", "Zoom Y follows Zoom X.");
    d.number("xfPosX", "Position X", 0, -100000, 100000, -4000, 4000, "xform", "Timeline pixels; positive moves the picture right.", 3);
    d.number("xfPosY", "Position Y", 0, -100000, 100000, -4000, 4000, "xform", "Timeline pixels; positive moves the picture up.", 3);
    d.number("xfRotation", "Rotation Angle", 0, -360, 360, -360, 360, "xform", "Degrees about the anchor point; positive turns the picture anticlockwise.", 3);
    d.number("xfAnchorX", "Anchor Point X", 0, -100000, 100000, -4000, 4000, "xform", "Timeline pixels from the frame centre: the point zoom and rotation turn about.", 3);
    d.number("xfAnchorY", "Anchor Point Y", 0, -100000, 100000, -4000, 4000, "xform", "Timeline pixels from the frame centre, up positive.", 3);
    d.number("xfPitch", "Pitch", 0, -89, 89, -89, 89, "xform", "Degrees: the picture tilted about its horizontal axis, in perspective (positive leans the top away).", 3);
    d.number("xfYaw", "Yaw", 0, -89, 89, -89, 89, "xform", "Degrees: the picture turned about its vertical axis, in perspective (positive turns the right side away).", 3);
    d.toggle("xfFlipH", "Flip Horizontal", false, "xform", "Mirror left and right.");
    d.toggle("xfFlipV", "Flip Vertical", false, "xform", "Mirror top and bottom.");
    d.group("raw", "Camera RAW", false, "Development controls (Resolve CinemaDNG panel layout); the native Camera RAW panel is bypassed.");
    d.toggle("developRaw", "Develop RAW", false, "raw",
             "On: the plug-in develops the original DNG frames itself, with the controls below. Off: the picture and the colour stay Resolve's own "
             "(its Camera RAW settings, your colour pipeline and PowerGrades as without the plug-in); the plug-in then only does the Transform, the "
             "stabilisation and the lens corrections on that picture. Off costs some sharpness (Resolve's picture at the timeline resolution is "
             "resampled once more); the vignette correction then needs Resolve Gamma set, and the controls below it do nothing.");
    const char* sg[] = {"Linear", "Gamma 2.2", "Gamma 2.4", "Gamma 2.6", "Rec.709", "sRGB", "Blackmagic Design Film", "Blackmagic Design 4K Film",
                        "Blackmagic Design 4.6K Film", "DaVinci Intermediate", "ACEScct"};
    d.choice("sourceGamma", "Resolve Gamma", sg, 11, 4, "raw",
             "Develop RAW off only, for the Vignette Correction: the Gamma set in Resolve's own Camera RAW panel for this clip. The vignette is removed in "
             "linear light, so the plug-in has to know how Resolve's picture is encoded. A wrong choice makes the corners too bright or too dark.");
    const char* sc[] = {"Scale to Fit", "Fill"};
    d.choice("sourceScaling", "Resolve Input Scaling", sc, 2, 0, "raw",
             "Develop RAW off only: how Resolve places a clip that has another shape than the timeline (Project Settings > Image Scaling > Mismatched "
             "resolution files, or the clip's own Input Scaling). Scale to Fit: whole, with bars. Fill: 'Scale full frame with crop'. Needed for the "
             "stabilisation and the lens corrections to line up; with those off the picture goes through unchanged either way.");
    const char* dq[] = {"Full Res. (RCD)", "Half Res."};
    d.choice("decodeQuality", "Decode Quality", dq, 2, 0, "raw", "Full: RCD demosaic at full resolution. Half: 2x2 binned, fastest.");
    d.choice("whiteBalance", "White Balance", WB_OPTIONS, 8, 0, "raw", "As Shot uses the camera's AsShotNeutral. Editing Color Temp or Tint switches to Custom.");
    d.choice("colorSpace", "Color Space", CS_OPTIONS, 6, 0, "raw", "Output primaries.");
    d.choice("gamma", "Gamma", GAMMA_OPTIONS, 7, 3, "raw", "Output encoding. Rec.709 has a linear toe; pure Gamma 2.2/2.4 make near-black noise sparkle. Use DaVinci Wide Gamut / DaVinci Intermediate in a DWG-managed project.");
    d.number("colorTemp", "Color Temp", 5600, 1500, 50000, 2000, 15000, "raw", "Kelvin (DNG colour calibration).", 0);
    d.number("tint", "Tint", 0, -150, 150, -150, 150, "raw", "Green-magenta balance.", 0);
    d.number("exposure", "Exposure", 0, -10, 10, -5, 5, "raw", "Stops, on top of the DNG baseline exposure.");
    d.number("sharpness", "Sharpness", 0, 0, 100, 0, 100, "raw", "Unsharp mask at source pitch.", 0);
    d.number("highlights", "Highlights", 0, -100, 100, -100, 100, "raw", "Negative recovers, positive brightens the upper stops.", 0);
    d.number("shadows", "Shadows", 0, -100, 100, -100, 100, "raw", "Positive lifts, negative deepens the lower stops.", 0);
    d.number("colorBoost", "Color Boost", 0, 0, 100, 0, 100, "raw", "Saturates low-saturation colours more than saturated ones.", 0);
    d.number("saturation", "Saturation", 50, 0, 100, 0, 100, "raw", "50 is neutral.", 0);
    d.number("midtones", "Midtones", 0, -100, 100, -100, 100, "raw", "Brightens or darkens the middle stops.", 0);
    d.number("lift", "Lift", 0, -100, 100, -100, 100, "raw", "Raises or lowers blacks in the output encoding.", 0);
    d.number("gain", "Gain", 0, -100, 100, -100, 100, "raw", "Scales the output encoding.", 0);
    d.number("contrast", "Contrast", 50, 0, 100, 0, 100, "raw", "50 is neutral; pivot at 18% grey.", 0);
    // Hidden since v1.3 (all off by default); kept so that projects which use them render as before.
    d.toggle("highlightRecovery", "Highlight Recovery", false, "raw", "Use unclipped channels above the green clip instead of clipping to white.");
    d.hide();
    d.toggle("gamutMapping", "Gamut Mapping", false, "raw", "Softly compress out-of-gamut colours towards the neutral axis.");
    d.hide();
    d.toggle("preToneCurve", "Apply Pre Tone Curve", false, "raw", "Soft highlight shoulder before the output encoding.");
    d.hide();
    d.toggle("softClip", "Apply Soft Clip", false, "raw", "Soft-clip encoded values above 0.8.");
    d.hide();
    // Hidden since v1.3.2, without a group (a group of hidden controls would still show its header):
    // Row Phase -0.125, Edge Anti-aliasing 80, Image Fit Scale to Fit. Kept so that projects load and render as before.
    d.number("rowPhase", "Row Phase (3K)", -0.125, -0.5, 0.5, -0.5, 0.5, nullptr,
             "3K only (sensor mode M98): re-spaces the G2/B rows that sit 0.5 px off the Bayer grid. -0.125 recommended; 0 = off.", 3);
    d.hide();
    d.number("deZigzag", "Edge Anti-aliasing (3K)", 80, 0, 100, 0, 100, nullptr,
             "3K only: removes the 2-row stair-steps on sharp near-horizontal edges left by the binned readout. 80 recommended; 0 = off.", 0);
    d.hide();
    const char* fit[] = {"Scale to Fit", "Fill", "Stretch", "Native 1:1"};
    d.choice("fit", "Image Fit", fit, 4, 0, nullptr, "How the DNG crop maps onto the timeline frame (Resolve's default input scaling is Scale to Fit).");
    d.hide();
    // Hidden: the kernel of the resampling. Best (Lanczos-3, widened when the picture is made
    // smaller) plays in real time; Fast (bilinear, as before 1.5.1) is kept for comparison.
    const char* resamplers[] = {"Best (Lanczos-3)", "Fast (bilinear)"};
    d.choice("resampling", "Resampling", resamplers, 2, 0, nullptr, "The kernel of the single resampling: Best (Lanczos-3) or Fast (bilinear).");
    d.hide();
    d.group("lens", "Lens Correction", false,
            "Corrections from the lens's own profile, which the camera writes into every frame. The Correction line says what the clip carries.");
    d.label("infoCorrection", "Correction", "lens", "What each of the two corrections uses for this clip: the picked file, the lens's own profile from the camera, or the installed Adobe profile.");
    d.toggle("lensShading", "Vignette Correction", false, "lens",
             "Removes the lens's vignette and colour shading. The vignette is only in the frames when the camera's Vignetting compensation "
             "was set to Auto. Applied to the raw data, before everything else.");
    d.toggle("lensDistortion", "Distortion Correction", false, "lens",
             "Straightens the lens's distortion, in the same single resampling as the fit, the Transform and the stabilisation "
             "(which then works on the straightened picture).");
    const char* fileHint = "Leave empty: the lens's own profile from the clip is used, and when the clip has none, the Adobe lens profile installed on "
                           "this computer for that lens. Or pick a profile file to use instead: a DNG shot with the same lens (a clip frame or a still), "
                           "an Adobe lens profile (.lcp) or a Lensfun file (.xml). The Correction line says what is used.";
    for (const char* const* f : {(const char* const[]){"lensShadingFile", "Vignette Profile"}, (const char* const[]){"lensDistortionFile", "Distortion Profile"}}) {
        auto p = d.define(kOfxParamTypeString, f[0], f[1], fileHint, "lens");
        set_s(p, kOfxParamPropStringMode, kOfxParamStringIsFilePath);
        set_s(p, kOfxParamPropDefault, "");
        set_i(p, kOfxParamPropAnimates, 0);
    }
    d.group("stab", "Stabilisation (gyro)", false,
            "Gyro stabilisation from the camera's gyro data: inside the DNG frames, or the <clip>.FPG file next to them. Found automatically; without gyro data this group does nothing.");
    d.label("stabStatus", "Gyro", "stab", "Gyro data in use (frames, sample rate, focal length, zoom), loading progress, or why stabilisation is off. Refreshes when a control is changed.");
    d.toggle("stabEnable", "Stabilisation", false, "stab", "Stabilise with the camera's gyro data (inside the DNG frames, or an .FPG file). Off by default.");
    d.number("stabSmoothness", "Smoothness", 0.1, 0, 30, 0, 3, "stab",
             "Time constant in seconds of the camera-path smoothing. 0 follows the camera (rolling-shutter correction only); large values lock the shot. More smoothing needs more zoom.");
    d.number("stabRollingShutter", "Rolling Shutter Correction", 1, 0, 1, 0, 1, "stab",
             "1 corrects each sensor row for the time it was read; 0 treats the frame as one instant.");
    d.number("stabReadout", "Rolling Shutter ms", 0, 0, 500, 0, 50, "stab",
             "Readout time of the sensor rows, first to last, in milliseconds. Filled in from the clip's gyro data; change it to override "
             "(the Gyro line then says override). 0 or reset = the clip's value again.", 2);
    d.number("stabSync", "Sync Offset (ms)", 0, -500, 500, -50, 50, "stab",
             "Added to the file's mark delay. Positive values use earlier gyro data for each frame. Normally 0.", 1);
    d.number("stabFocal", "Focal Length (mm)", 0, 0, 5000, 0, 200, "stab",
             "Filled in from the clip's gyro data; change it to override (the Gyro line then says override), for manual lenses or to trim the "
             "image scale. 0 or reset = the clip's value again.", 1);
    // Whether the two fields above show the clip's own value (kept up to date) or a value the user entered (never touched).
    d.toggle("stabFocalAuto", "Focal Length From Clip", true, "stab", "");
    d.hide();
    d.toggle("stabReadoutAuto", "Rolling Shutter ms From Clip", true, "stab", "");
    d.hide();
    d.toggle("stabAutoZoom", "Auto Zoom", true, "stab", "Zoom in by the smallest amount that hides the borders on every frame of the clip.");
    const char* ranges[] = {"Whole Clip", "Automatic", "Manual"};
    d.choice("stabRange", "Range", ranges, 3, 1, "stab",
             "The part of the clip that smoothing and zoom are worked out for. Automatic: the part this timeline clip uses, when the host reports it (the Gyro line shows what it got). "
             "Manual: the frames below. A shaky section outside the range no longer costs zoom inside it.");
    d.number("stabRangeStart", "Range Start (frame)", 1, 1, 10000000, 1, 10000, "stab", "Manual range: first source frame (the number in the DNG file name).", 0);
    d.number("stabRangeEnd", "Range End (frame)", 0, 0, 10000000, 0, 10000, "stab", "Manual range: last source frame (the number in the DNG file name); 0 = to the end of the clip.", 0);
    const char* zmodes[] = {"Fixed", "Dynamic"};
    d.choice("stabZoomMode", "Zoom Mode", zmodes, 2, 1, "stab",
             "Fixed: one zoom for the whole range. Dynamic: the zoom follows what each part needs and changes slowly, so a shaky section costs zoom only near itself.");
    d.number("stabZoomSmooth", "Smoothness", 4, 0.2, 30, 0.5, 10, "stab",
             "Zoom Mode Dynamic only (greyed out otherwise). Seconds over which the dynamic zoom looks ahead and eases. Larger = slower zoom changes, but more zoom around a shaky section.", 1);
    d.number("stabMaxZoom", "Auto Zoom Limit", 1.3, 1, 4, 1, 2, "stab",
             "Largest automatic zoom. Frames that would need more (violent moves) are stabilised less instead, so one jolt does not zoom the whole clip.", 2);
    d.number("stabZoom", "Zoom", 1, 0.5, 4, 0.8, 2, "stab", "Manual zoom, multiplies the automatic zoom.", 3);
    {
        auto p = d.define(kOfxParamTypeString, "gyroFile", "Gyro File", "Leave empty to use the gyro data in the DNGs, or the .FPG file next to them.", "stab");
        set_s(p, kOfxParamPropStringMode, kOfxParamStringIsFilePath);
        set_s(p, kOfxParamPropDefault, "");
        set_i(p, kOfxParamPropAnimates, 0);
        set_i(p, kOfxParamPropSecret, 1);   // hidden since v1.6: kept for projects that set it
    }
    // Hidden since v1.3.2, without a group: First DNG File (empty), Frame Mapping (Resolve Source Frame), Timeline Anchor (0).
    {
        auto p = d.define(kOfxParamTypeString, "sourceFile", "First DNG File",
                          "Leave empty: Resolve tells the plug-in which clip this is. Only if the Source line says the DNG could not be identified "
                          "(for example on the Fusion page), pick the clip's first DNG here.", nullptr);
        set_i(p, kOfxParamPropSecret, 1);
        set_s(p, kOfxParamPropStringMode, kOfxParamStringIsFilePath);
        set_s(p, kOfxParamPropDefault, "");
        set_i(p, kOfxParamPropAnimates, 0);
    }
    const char* fm[] = {"Resolve Source Frame", "Timeline Anchor", "Single File"};
    // Hidden since v1.3 (defaults: Resolve Source Frame, anchor 0); kept for projects that use them.
    d.choice("frameMode", "Frame Mapping", fm, 3, 0, nullptr, "Resolve Source Frame follows trims automatically on the Color page.");
    d.hide();
    d.number("anchor", "Timeline Anchor Frame", 0, -1e7, 1e7, 0, 100000, nullptr, "Timeline frame of the first DNG (Timeline Anchor mode).", 0);
    d.hide();
    return kOfxStatOK;
}

// ---- values ----
double dval(Instance& in, const char* n, double t) {
    double v = 0;
    gParam->paramGetValueAtTime(in.params.at(n), t, &v);
    return v;
}
int ival(Instance& in, const char* n, double t) {
    int v = 0;
    gParam->paramGetValueAtTime(in.params.at(n), t, &v);
    return v;
}
std::string sval(Instance& in, const char* n) {
    char* v = nullptr;
    gParam->paramGetValue(in.params.at(n), &v);
    return v ? v : "";
}

RawSettings settings(Instance& in, double t) {
    RawSettings s;
    s.decodeQuality = ival(in, "decodeQuality", t);
    s.whiteBalance = static_cast<WhiteBalance>(std::clamp(ival(in, "whiteBalance", t), 0, 7));
    s.colorSpace = static_cast<Primaries>(std::clamp(ival(in, "colorSpace", t), 0, 5));
    s.gamma = static_cast<Gamma>(std::clamp(ival(in, "gamma", t), 0, 6));
    s.colorTemp = dval(in, "colorTemp", t);
    s.tint = dval(in, "tint", t);
    s.exposure = dval(in, "exposure", t);
    s.sharpness = dval(in, "sharpness", t);
    s.highlights = dval(in, "highlights", t);
    s.shadows = dval(in, "shadows", t);
    s.colorBoost = dval(in, "colorBoost", t);
    s.saturation = dval(in, "saturation", t);
    s.midtones = dval(in, "midtones", t);
    s.lift = dval(in, "lift", t);
    s.gain = dval(in, "gain", t);
    s.contrast = dval(in, "contrast", t);
    s.highlightRecovery = ival(in, "highlightRecovery", t) != 0;
    s.gamutMapping = ival(in, "gamutMapping", t) != 0;
    s.preToneCurve = ival(in, "preToneCurve", t) != 0;
    s.softClip = ival(in, "softClip", t) != 0;
    s.rowPhase = dval(in, "rowPhase", t);
    s.deZigzag = dval(in, "deZigzag", t);
    s.fit = static_cast<Fit>(std::clamp(ival(in, "fit", t), 0, 3));   // hidden since 1.3.2: what older projects stored
    const int fitMode = ival(in, "fitMode", t);
    if (fitMode == 1) s.fit = Fit::FitWidth;
    else if (fitMode == 2) s.fit = Fit::FitHeight;
    s.resampler = ival(in, "resampling", t) == 1 ? kResampleBilinear : kResampleLanczos3;
    s.lensDistortion = ival(in, "lensDistortion", t) != 0;
    s.sourceFill = ival(in, "sourceScaling", t) == 1;
    s.xf.zoomX = dval(in, "xfZoomX", t);
    s.xf.zoomY = ival(in, "xfZoomLink", t) ? s.xf.zoomX : dval(in, "xfZoomY", t);
    s.xf.posX = dval(in, "xfPosX", t);
    s.xf.posY = dval(in, "xfPosY", t);
    s.xf.rotation = dval(in, "xfRotation", t);
    s.xf.anchorX = dval(in, "xfAnchorX", t);
    s.xf.anchorY = dval(in, "xfAnchorY", t);
    s.xf.pitch = dval(in, "xfPitch", t);
    s.xf.yaw = dval(in, "xfYaw", t);
    s.xf.flipH = ival(in, "xfFlipH", t) != 0;
    s.xf.flipV = ival(in, "xfFlipV", t) != 0;
    return s;
}

StabSettings stab_settings(Instance& in, double t) {
    StabSettings s;
    s.enable = ival(in, "stabEnable", t) != 0;
    s.smoothness = dval(in, "stabSmoothness", t);
    s.rollingShutter = dval(in, "stabRollingShutter", t);
    s.syncMs = dval(in, "stabSync", t);
    // The fields are prefilled with the clip's values; only a value the user entered overrides.
    s.focalMm = ival(in, "stabFocalAuto", t) ? 0.0 : dval(in, "stabFocal", t);
    s.readoutMs = ival(in, "stabReadoutAuto", t) ? 0.0 : dval(in, "stabReadout", t);
    s.autoZoom = ival(in, "stabAutoZoom", t) != 0;
    s.maxZoom = dval(in, "stabMaxZoom", t);
    s.zoom = dval(in, "stabZoom", t);
    s.zoomMode = ival(in, "stabZoomMode", t);
    {
        // Measurements only: SFP_GYRO_RAW_MARKS=1 takes each frame's time from its own mark.
        s.rawMarks = os::env("SFP_GYRO_RAW_MARKS") == "1";
    }
    s.zoomSmooth = dval(in, "stabZoomSmooth", t);
    // Used range.
    const int mode = ival(in, "stabRange", t);
    if (mode == 2) {
        const long long a = std::llround(dval(in, "stabRangeStart", t)), b = std::llround(dval(in, "stabRangeEnd", t));
        s.rangeFirst = std::max(0LL, a - 1);
        s.rangeLast = b > 0 ? b - 1 : 1000000000LL;
        s.rangeNote = "manual";
        if (b > 0 && b < a) { s.rangeLast = -1; s.rangeNote = "manual range is empty: whole clip used"; }
    } else if (mode == 1) {
        std::lock_guard<std::mutex> l(in.rangeM);
        char buf[200];
        if (!in.hostKnown) s.rangeNote = "automatic: known once a frame has been rendered";
        else if (in.hostRetimed) s.rangeNote = "automatic: the clip is retimed, whole clip used";
        else if (!in.hostHasRange) s.rangeNote = "automatic: the host reports no clip range";
        else {
            s.rangeFirst = std::llround(in.hostFirst + in.hostOffset);
            s.rangeLast = std::llround(in.hostLast + in.hostOffset);
            std::snprintf(buf, sizeof buf, "automatic: host clip range %.0f to %.0f, source frame = time %+.0f", in.hostFirst, in.hostLast, in.hostOffset + 1);
            s.rangeNote = buf;
            if (s.rangeLast < 0 || s.rangeLast < s.rangeFirst) { s.rangeLast = -1; s.rangeFirst = 0; }
            else s.rangeFirst = std::max(0LL, s.rangeFirst);
        }
    }
    return s;
}

// Range: Automatic. Remembers what the host says about the part of the source this timeline clip
// uses: the frame range of the Source clip (OFX: the frames for which the clip has images, in
// effect time) and, from this render, how effect time maps to source frames.
void note_host_range(Instance& in, double time, long long sourceFrame, const std::string& source) {
    OfxPropertySetHandle cp = nullptr;
    double r[2] = {0, 0};
    const bool has = gEffect->clipGetPropertySet(in.source, &cp) == kOfxStatOK && cp &&
                     gProp->propGetDoubleN(cp, kOfxImageEffectPropFrameRange, 2, r) == kOfxStatOK && r[1] > r[0];
    std::lock_guard<std::mutex> l(in.rangeM);
    const double offset = static_cast<double>(sourceFrame) - time;
    if (in.hostSource != source) { in.hostKnown = false; in.hostRetimed = false; in.hostSource = source; }
    if (in.hostKnown && std::fabs(offset - in.hostOffset) > 0.01) in.hostRetimed = true;   // not one source frame per frame
    // The range must contain this frame, or it is not a range in this effect's time.
    in.hostHasRange = has && time >= r[0] - 0.5 && time <= r[1] + 0.5;
    in.hostFirst = r[0];
    in.hostLast = r[1];
    in.hostOffset = offset;
    in.hostKnown = true;
}

std::string source_path(Instance& in) {
    std::string manual = sval(in, "sourceFile");
    if (!manual.empty()) return manual;
    return get_s(effect_props(in.handle), kOfxImageEffectPropSrcFilePath);
}

bool is_dng(const std::string& p) {
    if (p.size() < 4) return false;
    std::string e = p.substr(p.size() - 4);
    for (auto& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return e == ".dng";
}

// Reads metadata of the first frame to show as-shot values (outside render).
void sync_metadata(Instance& in, bool force) {
    std::string path = source_path(in);
    if (path.empty() || !is_dng(path)) {
        gParam->paramSetValue(in.params.at("info"), "No DNG source: apply Sigma fp RAW to the original CinemaDNG clip on the Color page.");
        in.haveMeta = false;
        return;
    }
    if (!force && path == in.metaPath) return;
    in.metaPath = path;
    std::vector<uint8_t> data;
    std::string err;
    DngInfo info;
    if (!read_file(path, data, err) || !parse_dng(data.data(), data.size(), info, err)) {
        gParam->paramSetValue(in.params.at("info"), ("Cannot read source: " + err).c_str());
        in.haveMeta = false;
        return;
    }
    in.meta = info;
    in.haveMeta = true;
    in.exposureS = in.exifFocalMm = 0;
    {
        ClipGeometry geo;
        if (dng_geometry(path, geo)) { in.exposureS = geo.exposureS; in.exifFocalMm = geo.focalMm; }
        // The sensor window is in the first frame's own gyro block: no need to wait for the clip's scan.
        in.windowW = in.windowH = 0;
        GyroBlock b;
        if (read_fpg2(path, b)) {
            GyroHeader gh;
            gh.activeW = info.cropW; gh.activeH = info.cropH;
            window_for(b.dcCrop, b.windowW, b.windowH, b.recordedW, b.recordedH, b.readoutUs, gh);
            in.windowW = gh.windowW; in.windowH = gh.windowH;
        }
    }
    gParam->paramSetValue(in.params.at("info"), path.substr(path.find_last_of("/\\") + 1).c_str());
}

// "3:2", "16:9", ... for a frame size, or "1.85:1".
std::string ratio_name(int w, int h) {
    if (w <= 0 || h <= 0) return "?";
    const double r = static_cast<double>(w) / h;
    static const struct { double r; const char* name; } known[] = {
        {1.0, "1:1"}, {4.0 / 3, "4:3"}, {1.5, "3:2"}, {16.0 / 9, "16:9"}, {256.0 / 135, "17:9"}, {2.0, "2:1"}, {2.39, "2.39:1"}, {2.4, "2.4:1"}};
    for (const auto& k : known)
        if (std::fabs(r / k.r - 1) < 0.006) return k.name;
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.2f:1", r);
    return buf;
}

// The rows of the Clip Info block below the first (outside render). clip and warp: the gyro
// data of the clip and the warp of the current frame, as gyro_for_frame gave them.
void sync_info(Instance& in, double t, const std::shared_ptr<const GyroClip>& clip, const StabParams& warp) {
    std::string exposure = "-", format = "-", lens = "-", correction = "-";
    char buf[512];
    if (in.haveMeta) {
        const DngInfo& m = in.meta;
        int rodW, rodH;
        double projW, projH;
        {
            std::lock_guard<std::mutex> l(in.rangeM);
            rodW = in.lastRodW; rodH = in.lastRodH; projW = in.lastProjW; projH = in.lastProjH;
        }
        // Exposure: shutter speed, angle, ISO, frame rate.
        std::string shutter = "shutter unknown";
        if (in.exposureS > 0) {
            const double d = 1.0 / in.exposureS;
            if (in.exposureS >= 0.5) std::snprintf(buf, sizeof buf, "%.1f s", in.exposureS);
            else if (std::fabs(d - std::round(d)) < 0.05) std::snprintf(buf, sizeof buf, "1/%.0f s", d);
            else std::snprintf(buf, sizeof buf, "1/%.1f s", d);
            shutter = buf;
            if (m.fps > 0) {
                std::snprintf(buf, sizeof buf, " | %.1f\xC2\xB0", 360.0 * in.exposureS * m.fps);
                shutter += buf;
            }
        }
        std::snprintf(buf, sizeof buf, "%s | ISO %d | %.3f fps", shutter.c_str(), m.iso, m.fps);
        exposure = buf;
        // Format: ratio, bit depth, sensor window, recorded size.
        const int windowW = clip && clip->h.windowW > 0 ? clip->h.windowW : in.windowW, windowH = clip && clip->h.windowW > 0 ? clip->h.windowH : in.windowH;
        if (windowW > 0 && windowH > 0)
            std::snprintf(buf, sizeof buf, "%s | %d-bit | sensor %dx%d > %dx%d", ratio_name(m.cropW, m.cropH).c_str(), m.bps, windowW, windowH, m.cropW, m.cropH);
        else
            std::snprintf(buf, sizeof buf, "%s | %d-bit | %dx%d (sensor window unknown)", ratio_name(m.cropW, m.cropH).c_str(), m.bps, m.cropW, m.cropH);
        format = buf;
        // Lens: focal length and the crop factor of what is seen across the frame's width.
        const bool override = !ival(in, "stabFocalAuto", t) && dval(in, "stabFocal", t) > 0;
        const double focal = override ? dval(in, "stabFocal", t) : clip && clip->h.focalMm > 0 ? clip->h.focalMm : in.exifFocalMm;
        if (focal > 0) std::snprintf(buf, sizeof buf, "%.1f mm%s", focal, override ? " (override)" : "");
        else std::snprintf(buf, sizeof buf, "focal length unknown");
        lens = buf;
        if (windowW > 0) {
            // Share of the recorded width that the frame shows: the fit, the Transform zoom, the stabiliser's zoom.
            const RawSettings s = settings(in, t);
            const double a = static_cast<double>(m.cropW) / std::max(1, m.cropH);
            const double c = projW > 0 && projH > 0 ? projW / projH : rodW > 0 && rodH > 0 ? static_cast<double>(rodW) / rodH : a;
            double zoom = 1;
            if (s.fit == Fit::FitHeight || s.fit == Fit::Fill) zoom = std::max(1.0, a / c);
            zoom *= std::fabs(s.xf.zoomX);
            const double stab = warp.on && warp.invZoom > 0 ? 1.0 / warp.invZoom : 1.0;
            zoom = std::max(1.0, zoom * stab);
            // Against the widest window the camera reads (6048 px, its 3:2 full-frame mode = 1.00x).
            const double crop = std::max(6048.0, static_cast<double>(windowW)) / windowW * zoom;
            std::snprintf(buf, sizeof buf, " | crop %.2fx", crop);
            lens += buf;
            if (focal > 0) { std::snprintf(buf, sizeof buf, " (= %.0f mm full frame)", focal * crop); lens += buf; }
        }
        // What each of the two corrections uses: the picked file, else what the frames carry, else the installed Adobe profile.
        std::string note;
        GainMap map;
        LensWarp warp;
        bool ownColour = true;
        if (!ival(in, "lensShading", t)) correction = "vignette: off";
        else {
            const bool have = lens_vignette(sval(in, "lensShadingFile"), m, map, ownColour, note);
            correction = "vignette: " + note;
            if (have && !ival(in, "developRaw", t)) correction += " (brightness only, on Resolve's picture)";
        }
        if (!ival(in, "lensDistortion", t)) correction += " | distortion: off";
        else { lens_distortion(sval(in, "lensDistortionFile"), m, warp, note); correction += " | distortion: " + note; }
    }
    const char* rows[4] = {"infoExposure", "infoFormat", "infoLens", "infoCorrection"};
    const std::string* text[4] = {&exposure, &format, &lens, &correction};
    const bool was = in.updating;
    in.updating = true;
    for (int k = 0; k < 4; ++k)
        if (sval(in, rows[k]) != *text[k]) gParam->paramSetValue(in.params.at(rows[k]), text[k]->c_str());
    in.updating = was;
}

// Shows which gyro file the clip uses, or why stabilisation is off (outside render).
void sync_stab_status(Instance& in, double t) {
    std::string status;
    std::shared_ptr<const GyroClip> clip;
    std::string path = source_path(in);
    StabParams warp{};
    // The source frame shown at time t, as far as the last render told.
    long long frame = 0;
    {
        std::lock_guard<std::mutex> l(in.rangeM);
        if (in.hostKnown) frame = std::max(0LL, std::llround(t + in.hostOffset));
    }
    if (path.empty() || !is_dng(path)) status = "Off: no DNG source";
    else {
        // In-frame gyro data is gathered in the background: give a short clip (or a cached one)
        // a moment, so that the text shows the result and not "Loading".
        for (int tries = 0; tries < 20; ++tries) {
            clip = gyro_for_frame(path, sval(in, "gyroFile"), frame, in.haveMeta ? in.meta.width : 0, in.haveMeta ? in.meta.height : 0,
                                  stab_settings(in, t), warp, status);
            if (status.compare(0, 7, "Loading") != 0) break;
            os::sleep_ms(15);
        }
    }
    // Prefill: the focal length and readout fields show the clip's own values unless the user
    // entered one. Only possible here (instance creation, parameter changes), not while rendering.
    if (clip) {
        const bool was = in.updating;
        in.updating = true;
        const double values[2] = {clip->h.focalMm, clip->h.readoutS * 1e3};
        const char* fields[2] = {"stabFocal", "stabReadout"};
        const char* autos[2] = {"stabFocalAuto", "stabReadoutAuto"};
        for (int k = 0; k < 2; ++k) {
            if (!ival(in, autos[k], t)) continue;
            const double shown = std::round(values[k] * 100) / 100;
            if (std::fabs(dval(in, fields[k], t) - shown) > 1e-9) gParam->paramSetValue(in.params.at(fields[k]), shown);
        }
        in.updating = was;
    }
    sync_info(in, t, clip, warp);
    if (status == in.stabStatus) return;
    in.stabStatus = status;
    const bool was = in.updating;
    in.updating = true;
    gParam->paramSetValue(in.params.at("stabStatus"), status.c_str());
    in.updating = was;
}

// Smoothness (of the zoom) only acts with Zoom Mode Dynamic: greyed out otherwise.
void sync_enabled(Instance& in, double t) {
    if (in.zoomSmoothProps) set_i(in.zoomSmoothProps, kOfxParamPropEnabled, ival(in, "stabZoomMode", t) == 1 ? 1 : 0);
}

void set_temp_tint(Instance& in, double temp, double tint) {
    in.updating = true;
    gParam->paramSetValue(in.params.at("colorTemp"), temp);
    gParam->paramSetValue(in.params.at("tint"), tint);
    in.updating = false;
}

OfxStatus create_instance(OfxImageEffectHandle h) {
    auto in = std::make_unique<Instance>();
    in->handle = h;
    gEffect->clipGetHandle(h, kOfxImageEffectOutputClipName, &in->output, nullptr);
    gEffect->clipGetHandle(h, kOfxImageEffectSimpleSourceClipName, &in->source, nullptr);
    OfxParamSetHandle ps = nullptr;
    gEffect->getParamSet(h, &ps);
    for (const char* n : {"info", "decodeQuality", "whiteBalance", "colorSpace", "gamma", "colorTemp", "tint", "exposure", "sharpness",
                          "highlights", "shadows", "colorBoost", "saturation", "midtones", "lift", "gain", "contrast",
                          "highlightRecovery", "gamutMapping", "preToneCurve", "softClip", "rowPhase",
                          "deZigzag", "fit", "sourceFile", "frameMode", "anchor", "stabStatus", "stabEnable", "stabSmoothness",
                          "stabRollingShutter", "stabSync", "stabFocal", "stabAutoZoom", "stabMaxZoom", "stabZoom", "gyroFile", "stabRange", "stabRangeStart", "stabRangeEnd",
                          "stabZoomMode", "stabZoomSmooth", "stabReadout", "stabFocalAuto", "stabReadoutAuto", "fitMode", "xfZoomX", "xfZoomY",
                          "xfZoomLink", "xfPosX", "xfPosY", "xfRotation", "xfAnchorX", "xfAnchorY", "xfPitch", "xfYaw", "xfFlipH", "xfFlipV", "resampling",
                          "infoExposure", "infoFormat", "infoLens", "infoCorrection", "lensShading", "lensDistortion", "lensShadingFile", "lensDistortionFile", "developRaw", "sourceGamma", "sourceScaling"}) {
        OfxParamHandle p = nullptr;
        OfxPropertySetHandle pp = nullptr;
        if (gParam->paramGetHandle(ps, n, &p, &pp) != kOfxStatOK) return kOfxStatFailed;
        in->params[n] = p;
        if (!std::strcmp(n, "stabZoomSmooth")) in->zoomSmoothProps = pp;
    }
    Instance* raw = in.release();
    gProp->propSetPointer(effect_props(h), kOfxPropInstanceData, 0, raw);
    try {
        sync_metadata(*raw, true);
        if (raw->haveMeta && ival(*raw, "whiteBalance", 0) == 0) {
            ColorSetup cs = color_setup(raw->meta, true, 0, 0, Primaries::Rec709);
            set_temp_tint(*raw, std::round(cs.temp), std::round(cs.tint));
        }
    } catch (...) {
    }
    try {
        sync_enabled(*raw, 0);
        sync_stab_status(*raw, 0);
    } catch (...) {
    }
    return kOfxStatOK;
}

OfxStatus destroy_instance(OfxImageEffectHandle h) {
    delete instance(h);
    gProp->propSetPointer(effect_props(h), kOfxPropInstanceData, 0, nullptr);
    return kOfxStatOK;
}

OfxStatus instance_changed(OfxImageEffectHandle h, OfxPropertySetHandle args) {
    Instance* in = instance(h);
    if (!in || in->updating) return kOfxStatReplyDefault;
    std::string name = get_s(args, kOfxPropName);
    std::string reason = get_s(args, kOfxPropChangeReason);
    double t = 0;
    gProp->propGetDouble(args, kOfxPropTime, 0, &t);
    if (name == "whiteBalance") {
        auto wb = static_cast<WhiteBalance>(ival(*in, "whiteBalance", t));
        double temp, tint;
        if (preset_temp_tint(wb, temp, tint)) set_temp_tint(*in, temp, tint);
        else if (wb == WhiteBalance::AsShot) {
            sync_metadata(*in, false);
            if (in->haveMeta) {
                ColorSetup cs = color_setup(in->meta, true, 0, 0, Primaries::Rec709);
                set_temp_tint(*in, std::round(cs.temp), std::round(cs.tint));
            }
        }
    } else if ((name == "colorTemp" || name == "tint") && reason == kOfxChangeUserEdited) {
        if (ival(*in, "whiteBalance", t) != static_cast<int>(WhiteBalance::Custom)) {
            in->updating = true;
            gParam->paramSetValue(in->params.at("whiteBalance"), static_cast<int>(WhiteBalance::Custom));
            in->updating = false;
        }
    } else if (name == "sourceFile" || name == kOfxImageEffectSimpleSourceClipName) {
        sync_metadata(*in, true);
        sync_stab_status(*in, t);
    } else if ((name == "xfZoomX" || name == "xfZoomY" || name == "xfZoomLink") && reason == kOfxChangeUserEdited) {
        // Linked zoom: show the same value in both fields.
        if (ival(*in, "xfZoomLink", t)) {
            in->updating = true;
            const char* other = name == "xfZoomY" ? "xfZoomX" : "xfZoomY";
            gParam->paramSetValue(in->params.at(other), dval(*in, name == "xfZoomY" ? "xfZoomY" : "xfZoomX", t));
            in->updating = false;
        }
        sync_stab_status(*in, t);   // the crop factor in Clip Info
    } else if (name == "fitMode" || name == "developRaw" || name == "sourceGamma" || name.compare(0, 4, "lens") == 0) {
        sync_stab_status(*in, t);
    } else if (name == "gyroFile" || name.compare(0, 4, "stab") == 0) {
        // A value typed into Focal Length or Rolling Shutter ms is the user's from now on; 0 (or
        // the control's reset) hands the field back to the clip's value.
        for (const char* field : {"stabFocal", "stabReadout"}) {
            if (name != field || reason != kOfxChangeUserEdited) continue;
            in->updating = true;
            gParam->paramSetValue(in->params.at(name == "stabFocal" ? "stabFocalAuto" : "stabReadoutAuto"), dval(*in, field, t) > 0 ? 0 : 1);
            in->updating = false;
        }
        if (name == "stabZoomMode") sync_enabled(*in, t);
        if (name != "stabStatus") sync_stab_status(*in, t);
    }
    return kOfxStatOK;
}

OfxStatus clip_preferences(OfxImageEffectHandle h, OfxPropertySetHandle out) {
    (void)h;
    set_s(out, kOfxImageEffectPropPreMultiplication, kOfxImageOpaque);
    set_i(out, kOfxImageEffectFrameVarying, 1);
    return kOfxStatOK;
}

// The node's frame as the timeline frame: asked for only with Fit Width / Fit Height, which need
// the room beyond the clip's own frame, and only when that frame differs from the timeline's.
// With Scale to Fit (the default) the host's own frame stands: a clip of another shape than the
// timeline must come out exactly where the host puts it.
OfxStatus region_of_definition(OfxImageEffectHandle h, OfxPropertySetHandle args, OfxPropertySetHandle out) {
    Instance* in = instance(h);
    if (!in) return kOfxStatReplyDefault;
    double time = 0, size[2] = {0, 0}, off[2] = {0, 0};
    gProp->propGetDouble(args, kOfxPropTime, 0, &time);
    if (ival(*in, "fitMode", time) == 0) return kOfxStatReplyDefault;
    if (gProp->propGetDoubleN(effect_props(h), kOfxImageEffectPropProjectSize, 2, size) != kOfxStatOK || !(size[0] > 0 && size[1] > 0)) return kOfxStatReplyDefault;
    gProp->propGetDoubleN(effect_props(h), kOfxImageEffectPropProjectOffset, 2, off);
    OfxRectD src{};
    if (gEffect->clipGetRegionOfDefinition(in->source, time, &src) != kOfxStatOK) return kOfxStatReplyDefault;
    const double want[4] = {off[0], off[1], off[0] + size[0], off[1] + size[1]};
    if (std::fabs(src.x1 - want[0]) < 0.5 && std::fabs(src.y1 - want[1]) < 0.5 && std::fabs(src.x2 - want[2]) < 0.5 && std::fabs(src.y2 - want[3]) < 0.5)
        return kOfxStatReplyDefault;
    gProp->propSetDoubleN(out, kOfxImageEffectPropRegionOfDefinition, 4, want);
    return kOfxStatOK;
}

// Numbered path of the requested source frame.
bool frame_path(Instance& in, OfxPropertySetHandle args, double time, std::string& path, long long& number,
                std::string& prefix, int& digits, std::string& suffix, std::string& err) {
    std::string first = source_path(in);
    if (first.empty()) { err = "The original DNG could not be identified: apply Sigma fp RAW to the original DNG clip on the Color page."; return false; }
    if (!is_dng(first)) { err = "The source is not a DNG sequence (" + first + ")."; return false; }
    int mode = ival(in, "frameMode", time);
    long long base;
    if (!split_sequence(first, prefix, base, digits, suffix)) {
        path = first; number = 0; prefix.clear();
        return true;
    }
    long long offset = 0;
    if (mode == 2) offset = 0;
    else if (mode == 1) {
        if (sval(in, "sourceFile").empty()) { err = "Timeline Anchor needs a First DNG File."; return false; }
        offset = static_cast<long long>(std::floor(time - dval(in, "anchor", time) + 0.5));
    } else {
        int f = 0;
        if (gProp->propGetInt(args, kOfxImageEffectPropSrcFrame, 0, &f) == kOfxStatOK) {
            offset = f;
        } else if (!sval(in, "sourceFile").empty()) {
            // No source frame from the host (e.g. the Fusion page) but the first DNG is given:
            // the frame follows the effect time from the anchor (0 unless an older project set it).
            offset = static_cast<long long>(std::floor(time - dval(in, "anchor", time) + 0.5));
        } else {
            err = "Resolve did not supply the source frame here (e.g. Fusion page): apply Sigma fp RAW on the Color page.";
            return false;
        }
    }
    number = base + offset;
    if (number < 0) { err = "Frame before the start of the sequence."; return false; }
    path = sequence_path(prefix, number, digits, suffix);
    return true;
}

OfxStatus render(OfxImageEffectHandle h, OfxPropertySetHandle args) {
    Instance* in = instance(h);
    if (!in) return kOfxStatFailed;
    double time = 0;
    gProp->propGetDouble(args, kOfxPropTime, 0, &time);
    OfxPropertySetHandle img = nullptr;
    if (gEffect->clipGetImage(in->output, time, nullptr, &img) != kOfxStatOK || !img) return kOfxStatFailed;
    struct Release { OfxPropertySetHandle p; ~Release() { gEffect->clipReleaseImage(p); } } release{img};
    void* data = nullptr;
    int bounds[4] = {}, rod[4] = {}, rowBytes = 0;
    gProp->propGetPointer(img, kOfxImagePropData, 0, &data);
    gProp->propGetIntN(img, kOfxImagePropBounds, 4, bounds);
    if (gProp->propGetIntN(img, kOfxImagePropRegionOfDefinition, 4, rod) != kOfxStatOK) std::memcpy(rod, bounds, sizeof rod);
    gProp->propGetInt(img, kOfxImagePropRowBytes, 0, &rowBytes);
    if (get_s(img, kOfxImageEffectPropPixelDepth) != kOfxBitDepthFloat || get_s(img, kOfxImageEffectPropComponents) != kOfxImageComponentRGBA) {
        post_error(h, "Sigma fp RAW needs 32-bit float RGBA images.");
        return kOfxStatErrFormat;
    }
    int cudaEnabled = 0;
    gProp->propGetInt(args, kOfxImageEffectPropCudaEnabled, 0, &cudaEnabled);
    void* stream = nullptr;
    if (cudaEnabled) gProp->propGetPointer(args, kOfxImageEffectPropCudaStream, 0, &stream);
    double scale[2] = {1, 1};
    gProp->propGetDoubleN(args, kOfxImageEffectPropRenderScale, 2, scale);

    std::lock_guard<std::mutex> lock(in->m);
    std::string path, prefix, suffix, err;
    long long number = 0;
    int digits = 0;
    if (!frame_path(*in, args, time, path, number, prefix, digits, suffix, err)) { post_error(h, err); return kOfxStatFailed; }
    const bool developRaw = ival(*in, "developRaw", time) != 0;
    const bool wantShading = ival(*in, "lensShading", time) != 0;
    const bool shading = developRaw && wantShading;
    const std::string vignetteFile = shading ? sval(*in, "lensShadingFile") : std::string();
    auto frame = FrameCache::get().fetch(path, err, shading, vignetteFile);
    std::string substitute;
    if (!frame && !prefix.empty() && err.compare(0, 11, "Cannot open") != 0 && err.compare(0, 11, "Read failed") != 0) {
        // The file is there but its picture is damaged (seen in camera files: a corrupt LJ92 tile).
        // Show the nearest earlier frame that decodes instead of failing, and say so.
        for (long long k = number - 1; k >= std::max(0LL, number - 10) && !frame; --k) {
            std::string e2;
            frame = FrameCache::get().fetch(sequence_path(prefix, k, digits, suffix), e2, shading, vignetteFile);
            if (frame) {
                char buf[512];
                std::snprintf(buf, sizeof buf, "Sigma fp RAW: frame %lld cannot be decoded (%s): the file is damaged; showing frame %lld instead.",
                              number, err.c_str(), k);
                substitute = buf;
                number = k;
                path = sequence_path(prefix, k, digits, suffix);
            }
        }
    }
    if (!frame) { post_error(h, "Sigma fp RAW: " + err); return kOfxStatFailed; }
    // Read ahead in the playback direction.
    if (!prefix.empty()) {
        if (in->lastFrame >= 0 && number != in->lastFrame) in->direction = number > in->lastFrame ? 1 : -1;
        in->lastFrame = number;
        std::vector<std::string> ahead;
        for (int k = 1; k <= 40; ++k) {
            long long n = number + in->direction * k;
            if (n >= 0) ahead.push_back(sequence_path(prefix, n, digits, suffix));
        }
        FrameCache::get().prefetch(ahead, shading, vignetteFile);
    }
    RawSettings s = settings(*in, time);
    if (s.lensDistortion) {
        std::string note;
        lens_distortion(sval(*in, "lensDistortionFile"), frame->info, s.lensProfile, note);
    }
    if (!developRaw && wantShading) {
        // Resolve's picture: the vignette's brightness is removed from it, in linear light.
        std::string note;
        bool ownColour = true;
        if (!lens_vignette(sval(*in, "lensShadingFile"), frame->info, s.sourceVignette, ownColour, note)) s.sourceVignette = GainMap{};
        s.sourceGamma = ival(*in, "sourceGamma", time);
    }
    {
        std::string why;
        s.binned = frame_binning(path, frame->info.width, frame->info.height, why);
    }
    s.renderScaleX = scale[0];
    s.renderScaleY = scale[1];
    {
        // Gyro stabilisation: the camera numbers a clip's DNGs from 1, the gyro file's frames from 0.
        std::string status;
        if (!prefix.empty() && ival(*in, "frameMode", time) != 2) note_host_range(*in, time, number - 1, prefix);
        gyro_for_frame(path, sval(*in, "gyroFile"), prefix.empty() ? 0 : number - 1, frame->info.width, frame->info.height,
                       stab_settings(*in, time), s.stab, status);
    }
    Target t;
    t.width = bounds[2] - bounds[0];
    t.height = bounds[3] - bounds[1];
    t.rowBytes = rowBytes;
    t.boundsX = bounds[0] - rod[0];
    t.boundsY = bounds[1] - rod[1];
    t.rodW = rod[2] - rod[0];
    t.rodH = rod[3] - rod[1];
    {
        // The timeline frame. When the node's frame has another shape (the host hands over the
        // clip's own frame and fits it into the timeline afterwards), Fit Width / Fit Height
        // still mean the timeline frame: its size in this frame's pixels, centred on it.
        double proj[2] = {0, 0};
        if (gProp->propGetDoubleN(effect_props(h), kOfxImageEffectPropProjectSize, 2, proj) != kOfxStatOK || !(proj[0] > 0 && proj[1] > 0)) proj[0] = proj[1] = 0;
        proj[0] *= scale[0]; proj[1] *= scale[1];
        if (proj[0] > 0 && t.rodW > 0 && t.rodH > 0) {
            const double shape = (proj[0] / proj[1]) / (static_cast<double>(t.rodW) / t.rodH);
            if (std::fabs(shape - 1) > 0.004) {
                const double k = std::min(proj[0] / t.rodW, proj[1] / t.rodH);   // how the host fits this frame into the timeline
                t.compW = proj[0] / k;
                t.compH = proj[1] / k;
            }
        }
        std::lock_guard<std::mutex> l(in->rangeM);
        in->lastRodW = t.rodW; in->lastRodH = t.rodH;
        in->lastProjW = proj[0]; in->lastProjH = proj[1];
    }
    if (cudaEnabled) t.device = reinterpret_cast<CUdeviceptr>(data);
    else t.host = data;
    // Develop RAW off: Resolve's own picture of the clip is the input.
    SourceImage source;
    OfxPropertySetHandle srcImg = nullptr;
    struct ReleaseSource { OfxPropertySetHandle& p; ~ReleaseSource() { if (p) gEffect->clipReleaseImage(p); } } releaseSource{srcImg};
    if (!developRaw) {
        if (gEffect->clipGetImage(in->source, time, nullptr, &srcImg) != kOfxStatOK || !srcImg) {
            srcImg = nullptr;
            post_error(h, "Sigma fp RAW: the host did not supply the clip's picture (needed with Develop RAW off).");
            return kOfxStatFailed;
        }
        void* srcData = nullptr;
        int sb[4] = {}, sr[4] = {};
        gProp->propGetPointer(srcImg, kOfxImagePropData, 0, &srcData);
        gProp->propGetIntN(srcImg, kOfxImagePropBounds, 4, sb);
        if (gProp->propGetIntN(srcImg, kOfxImagePropRegionOfDefinition, 4, sr) != kOfxStatOK) std::memcpy(sr, sb, sizeof sr);
        gProp->propGetInt(srcImg, kOfxImagePropRowBytes, 0, &source.rowBytes);
        if (!srcData || get_s(srcImg, kOfxImageEffectPropPixelDepth) != kOfxBitDepthFloat || get_s(srcImg, kOfxImageEffectPropComponents) != kOfxImageComponentRGBA) {
            post_error(h, "Sigma fp RAW needs 32-bit float RGBA images (Develop RAW off).");
            return kOfxStatErrFormat;
        }
        if (cudaEnabled) source.device = reinterpret_cast<CUdeviceptr>(srcData);
        else source.host = srcData;
        source.width = sb[2] - sb[0];
        source.height = sb[3] - sb[1];
        source.offX = sb[0] - sr[0];
        source.offTop = sr[3] - sb[3];
        source.rodW = sr[2] - sr[0];
        source.rodH = sr[3] - sr[1];
    }
    if (!Developer::get().develop(*frame, s, static_cast<CUstream>(stream), t, err, nullptr, developRaw ? nullptr : &source)) {
        post_error(h, "Sigma fp RAW: " + err);
        return kOfxStatFailed;
    }
    if (substitute.empty()) clear_error(h);
    else if (gMsg2) gMsg2->setPersistentMessage(h, kOfxMessageWarning, "sfp_warning", "%s", substitute.c_str());
    else if (gMsg1) gMsg1->message(h, kOfxMessageWarning, "sfp_warning", "%s", substitute.c_str());
    return kOfxStatOK;
}

OfxStatus main_entry(const char* action, const void* handle, OfxPropertySetHandle in, OfxPropertySetHandle out) {
    auto h = static_cast<OfxImageEffectHandle>(const_cast<void*>(handle));
    try {
        if (!std::strcmp(action, kOfxActionLoad)) {
            if (!gHost) return kOfxStatErrMissingHostFeature;
            gProp = static_cast<const OfxPropertySuiteV1*>(gHost->fetchSuite(gHost->host, kOfxPropertySuite, 1));
            gEffect = static_cast<const OfxImageEffectSuiteV1*>(gHost->fetchSuite(gHost->host, kOfxImageEffectSuite, 1));
            gParam = static_cast<const OfxParameterSuiteV1*>(gHost->fetchSuite(gHost->host, kOfxParameterSuite, 1));
            gMsg2 = static_cast<const OfxMessageSuiteV2*>(gHost->fetchSuite(gHost->host, kOfxMessageSuite, 2));
            gMsg1 = static_cast<const OfxMessageSuiteV1*>(gHost->fetchSuite(gHost->host, kOfxMessageSuite, 1));
            return gProp && gEffect && gParam ? kOfxStatOK : kOfxStatErrMissingHostFeature;
        }
        if (!std::strcmp(action, kOfxActionUnload)) { gyro_shutdown(); return kOfxStatOK; }
        if (!std::strcmp(action, kOfxActionDescribe)) return describe(h);
        if (!std::strcmp(action, kOfxImageEffectActionDescribeInContext)) return describe_in_context(h);
        if (!std::strcmp(action, kOfxActionCreateInstance)) return create_instance(h);
        if (!std::strcmp(action, kOfxActionDestroyInstance)) return destroy_instance(h);
        if (!std::strcmp(action, kOfxActionInstanceChanged)) return instance_changed(h, in);
        if (!std::strcmp(action, kOfxImageEffectActionGetClipPreferences)) return clip_preferences(h, out);
        if (!std::strcmp(action, kOfxImageEffectActionRender)) return render(h, in);
        if (!std::strcmp(action, kOfxImageEffectActionGetRegionOfDefinition)) return region_of_definition(h, in, out);
        if (!std::strcmp(action, kOfxImageEffectActionIsIdentity)) return kOfxStatReplyDefault;
        if (!std::strcmp(action, kOfxImageEffectActionBeginSequenceRender) || !std::strcmp(action, kOfxImageEffectActionEndSequenceRender))
            return kOfxStatOK;
    } catch (const std::exception& e) {
        if (h) post_error(h, std::string("Sigma fp RAW: ") + e.what());
        return kOfxStatFailed;
    } catch (...) {
        return kOfxStatFailed;
    }
    return kOfxStatReplyDefault;
}

void set_host(OfxHost* host) { gHost = host; }

OfxPlugin gPlugin = {kOfxImageEffectPluginApi, 1, "com.sigmafpmods.raw", 1, 7, set_host, main_entry};

}  // namespace

// The two entry points a host looks up; everything else stays private to the library.
#ifdef _WIN32
#define SFP_EXPORT __declspec(dllexport)
#else
#define SFP_EXPORT __attribute__((visibility("default")))
#endif
extern "C" {
SFP_EXPORT int OfxGetNumberOfPlugins(void) { return 1; }
SFP_EXPORT OfxPlugin* OfxGetPlugin(int i) { return i == 0 ? &gPlugin : nullptr; }
}
