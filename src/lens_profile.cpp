// Lens correction profiles from a file, for the Lens Correction group's two file fields:
//   - a DNG (a frame of a clip or a still of the same lens): its GainMap and WarpRectilinear opcodes,
//   - an Adobe lens profile (.lcp): PerspectiveModel (radial terms) and VignetteModel,
//   - a Lensfun database file (.xml): distortion ptlens / poly3 / poly5, vignetting pa.
// Every profile is brought to the same two forms the camera's own profile has: a brightness map
// over the whole sensor (GainMap) and a radial polynomial with its radius in sensor pixels
// (LensWarp). Profiles made with another camera are scaled by their sensor size.
#include "lens_profile.h"
#include "platform.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>

namespace sfp {
namespace {

const double kSensorW = 6048, kSensorH = 4032;   // the whole sensor, pixels
const double kPitchMm = 35.9 / 6000;             // pixel pitch of the fp's sensor

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::string file_name(const std::string& p) { return p.substr(p.find_last_of("/\\") + 1); }

// One calibration point of a profile.
struct Entry {
    double focal = 0, aperture = 0, distance = 0;   // 0 = not given
    bool hasWarp = false, hasVignette = false;
    double c[7] = {1, 0, 0, 0, 0, 0, 0};            // distortion: factor(r) = sum c[i] r^i, corrected -> recorded
    double warpRadiusMm = 0;                        // r = 1 at this distance from the centre
    double v[3] = {0, 0, 0};                        // vignette falloff 1 + v0 r^2 + v1 r^4 + v2 r^6
    double vignetteRadiusMm = 0;
};

struct Lens {
    std::string name;
    double formatFactor = 1;     // sensor of the calibration camera against full frame
    std::vector<Entry> entries;
};

struct Profile {
    std::string kind;            // "DNG", "Adobe profile", "Lensfun"
    std::string error;           // not empty: the file cannot be used
    std::vector<Lens> lenses;    // Adobe: one; Lensfun: every lens of the file
    DngInfo dng;                 // kind DNG
};

// ---- XML, as far as these two formats need it ----

// Value of name="..." or <name>...</name> in text; false when absent.
bool xml_value(const std::string& text, const std::string& name, std::string& out) {
    size_t at = 0;
    while ((at = text.find(name, at)) != std::string::npos) {
        const size_t end = at + name.size();
        const char before = at ? text[at - 1] : ' ';
        if (end < text.size() && text[end] == '=' && (before == ' ' || before == '\n' || before == '\t' || before == '\r')) {
            const char q = end + 1 < text.size() ? text[end + 1] : 0;
            if (q == '"' || q == '\'') {
                const size_t close = text.find(q, end + 2);
                if (close != std::string::npos) { out = text.substr(end + 2, close - end - 2); return true; }
            }
        } else if (before == '<' && end < text.size() && text[end] == '>') {
            const size_t close = text.find("</" + name + ">", end);
            if (close != std::string::npos) { out = text.substr(end + 1, close - end - 1); return true; }
        }
        at = end;
    }
    return false;
}
double xml_number(const std::string& text, const std::string& name, double def = 0) {
    std::string v;
    if (!xml_value(text, name, v)) return def;
    char* end = nullptr;
    const double d = std::strtod(v.c_str(), &end);
    return end == v.c_str() ? def : d;
}

// ---- Adobe .lcp ----
// Each calibration point is a PerspectiveModel (rectilinear lenses) with, nested in it, a
// VignetteModel. Coordinates are in units of the focal length: FocalLengthX times the longer
// image side, or, when it is not given, focal length x SensorFormatFactor / 35 of that side
// (the longer side being 36 mm / SensorFormatFactor).
void parse_lcp(const std::string& text, Profile& p) {
    p.kind = "Adobe profile";
    Lens lens;
    xml_value(text, "stCamera:LensPrettyName", lens.name) || xml_value(text, "stCamera:Lens", lens.name);
    const std::string open = "<stCamera:PerspectiveModel", close = "</stCamera:PerspectiveModel>";
    size_t at = 0, previous = 0;
    while ((at = text.find(open, at)) != std::string::npos) {
        size_t end = text.find(close, at);
        if (end == std::string::npos) end = text.size();
        // The point's own values sit in the description that holds the model.
        const std::string head = text.substr(previous, at - previous);
        const size_t desc = head.rfind(close);   // after the previous point's model
        const std::string own = desc == std::string::npos ? head : head.substr(desc);
        const std::string body = text.substr(at, end - at);
        const size_t vAt = body.find("<stCamera:VignetteModel");
        size_t nested = body.size();
        for (const char* n : {"<stCamera:VignetteModel", "<stCamera:ChromaticRedGreenModel", "<stCamera:ChromaticGreenModel", "<stCamera:ChromaticBlueGreenModel"})
            nested = std::min(nested, body.find(n));
        const std::string model = body.substr(0, nested);
        Entry e;
        e.focal = xml_number(own, "stCamera:FocalLength");
        // ApertureValue is the APEX value: f-number = sqrt(2)^value.
        std::string av;
        if (xml_value(own, "stCamera:ApertureValue", av)) e.aperture = std::pow(2.0, 0.5 * std::atof(av.c_str()));
        e.distance = xml_number(own, "stCamera:FocusDistance");
        const double sff = std::max(0.1, xml_number(own, "stCamera:SensorFormatFactor", xml_number(text, "stCamera:SensorFormatFactor", 1)));
        const double longMm = 36.0 / sff;
        lens.formatFactor = sff;
        auto radius = [&](const std::string& scope) {
            const double fx = xml_number(scope, "stCamera:FocalLengthX", -1);
            return fx > 0 ? fx * longMm : e.focal * sff / 35.0 * longMm;
        };
        std::string unused;
        if (xml_value(model, "stCamera:RadialDistortParam1", unused)) {
            e.hasWarp = true;
            e.c[2] = xml_number(model, "stCamera:RadialDistortParam1");
            e.c[4] = xml_number(model, "stCamera:RadialDistortParam2");
            e.c[6] = xml_number(model, "stCamera:RadialDistortParam3");
            e.warpRadiusMm = radius(model);
        }
        if (vAt != std::string::npos) {
            size_t vEnd = body.find("<stCamera:VignetteModelPiecewiseParam", vAt);
            if (vEnd == std::string::npos) vEnd = body.find("</stCamera:VignetteModel>", vAt);
            const std::string v = body.substr(vAt, vEnd == std::string::npos ? std::string::npos : vEnd - vAt);
            if (xml_value(v, "stCamera:VignetteModelParam1", unused)) {
                e.hasVignette = true;
                e.v[0] = xml_number(v, "stCamera:VignetteModelParam1");
                e.v[1] = xml_number(v, "stCamera:VignetteModelParam2");
                e.v[2] = xml_number(v, "stCamera:VignetteModelParam3");
                e.vignetteRadiusMm = radius(v);
                if (!(e.vignetteRadiusMm > 0)) e.vignetteRadiusMm = radius(model);
            }
        }
        if ((e.hasWarp && e.warpRadiusMm > 0) || (e.hasVignette && e.vignetteRadiusMm > 0)) lens.entries.push_back(e);
        previous = at;
        at = end;
    }
    if (lens.entries.empty()) {
        p.error = text.find("<stCamera:FisheyeModel") != std::string::npos ? "fisheye profiles are not supported" : "no lens model found in the profile";
        return;
    }
    p.lenses.push_back(std::move(lens));
}

// ---- Lensfun .xml ----
// Distortion (Hugin's convention): recorded radius = corrected radius x factor, r = 1 at half
// the shorter side of the calibration camera's frame. Vignetting "pa": r = 1 at its corner.
void parse_lensfun(const std::string& text, Profile& p) {
    p.kind = "Lensfun";
    size_t at = 0;
    while ((at = text.find("<lens>", at)) != std::string::npos) {
        size_t end = text.find("</lens>", at);
        if (end == std::string::npos) end = text.size();
        const std::string l = text.substr(at, end - at);
        at = end;
        Lens lens;
        // The plain <model>, not a translated <model lang="..">.
        xml_value(l, "model", lens.name);
        const double crop = std::max(0.1, xml_number(l, "cropfactor", 1));
        lens.formatFactor = crop;
        double aspect = 1.5;
        std::string ar;
        if (xml_value(l, "aspect-ratio", ar)) {
            const size_t colon = ar.find(':');
            const double a = std::atof(ar.c_str()), b = colon == std::string::npos ? 1.0 : std::atof(ar.c_str() + colon + 1);
            if (a > 0 && b > 0) aspect = std::max(a, b) / std::min(a, b);
        }
        const double halfDiag = 0.5 * std::sqrt(36.0 * 36.0 + 24.0 * 24.0) / crop;
        const double halfShort = halfDiag / std::sqrt(1 + aspect * aspect);
        size_t t = 0;
        while ((t = l.find('<', t)) != std::string::npos) {
            const size_t close = l.find('>', t);
            if (close == std::string::npos) break;
            const std::string tag = l.substr(t, close - t);
            t = close;
            const bool dist = tag.compare(0, 12, "<distortion ") == 0, vig = tag.compare(0, 12, "<vignetting ") == 0;
            if (!dist && !vig) continue;
            std::string model;
            xml_value(tag, "model", model);
            Entry e;
            e.focal = xml_number(tag, "focal");
            if (dist) {
                const double a = xml_number(tag, "a"), b = xml_number(tag, "b"), c = xml_number(tag, "c");
                const double k1 = xml_number(tag, "k1"), k2 = xml_number(tag, "k2");
                if (model == "ptlens") { e.c[0] = 1 - a - b - c; e.c[1] = c; e.c[2] = b; e.c[3] = a; }
                else if (model == "poly3") { e.c[0] = 1 - k1; e.c[2] = k1; }
                else if (model == "poly5") { e.c[2] = k1; e.c[4] = k2; }
                else continue;
                e.hasWarp = true;
                e.warpRadiusMm = halfShort;
            } else {
                if (model != "pa") continue;
                e.aperture = xml_number(tag, "aperture");
                e.distance = xml_number(tag, "distance");
                e.v[0] = xml_number(tag, "k1"); e.v[1] = xml_number(tag, "k2"); e.v[2] = xml_number(tag, "k3");
                e.hasVignette = true;
                e.vignetteRadiusMm = halfDiag;
            }
            lens.entries.push_back(e);
        }
        if (!lens.entries.empty()) p.lenses.push_back(std::move(lens));
    }
    if (p.lenses.empty()) p.error = "no lens calibration found in the file";
}

std::shared_ptr<const Profile> load(const std::string& file) {
    static std::mutex m;
    static std::map<std::string, std::shared_ptr<const Profile>> cache;
    {
        std::lock_guard<std::mutex> l(m);
        auto it = cache.find(file);
        if (it != cache.end()) return it->second;
    }
    auto p = std::make_shared<Profile>();
    std::vector<uint8_t> data;
    std::string err;
    if (!read_file(file, data, err)) p->error = "cannot open the file";
    else if (data.size() > 8 && ((data[0] == 'I' && data[1] == 'I') || (data[0] == 'M' && data[1] == 'M'))) {
        p->kind = "DNG";
        if (!parse_dng(data.data(), data.size(), p->dng, err)) p->error = "not a DNG that can be read (" + err + ")";
    } else {
        const std::string text(data.begin(), data.end());
        if (text.find("stCamera:") != std::string::npos) parse_lcp(text, *p);
        else if (text.find("<lensdatabase") != std::string::npos || text.find("<lens>") != std::string::npos) parse_lensfun(text, *p);
        else p->error = "unknown format (use a DNG, an Adobe .lcp or a Lensfun .xml)";
    }
    std::lock_guard<std::mutex> l(m);
    if (cache.size() > 16) cache.clear();
    cache[file] = p;
    return p;
}

// The words of a lens name, lower case; "F4.5" and "f/4.5" both give "4.5".
std::vector<std::string> words(const std::string& s) {
    std::vector<std::string> out;
    std::string w;
    for (char ch : lower(s) + " ") {
        if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '.' || ch == '-') w += ch;
        else if (!w.empty()) {
            if (w.size() > 1 && w[0] == 'f' && std::isdigit(static_cast<unsigned char>(w[1]))) w = w.substr(1);
            out.push_back(w);
            w.clear();
        }
    }
    return out;
}

// The lens of a Lensfun file that the clip's lens name fits best (words in common).
const Lens* pick_lens(const Profile& p, const DngInfo& clip, std::string& why) {
    if (p.lenses.size() == 1) return &p.lenses[0];
    const auto want = words(clip.lensModel);
    const Lens* best = nullptr;
    double bestScore = 0;
    for (const auto& l : p.lenses) {
        const auto have = words(l.name);
        int common = 0;
        for (const auto& w : want)
            if (std::find(have.begin(), have.end(), w) != have.end()) ++common;
        // The focal length must be among the common words; then the share of matching words counts.
        bool focal = false;
        for (const auto& w : have)
            if (w.find("mm") != std::string::npos && std::find(want.begin(), want.end(), w) != want.end()) focal = true;
        const double score = focal ? common / std::sqrt(static_cast<double>(std::max<size_t>(1, have.size()) * std::max<size_t>(1, want.size()))) : 0;
        if (score > bestScore) { bestScore = score; best = &l; }
    }
    if (!best || bestScore < 0.5) {
        why = clip.lensModel.empty() ? "the clip does not name its lens" : "no lens like \"" + clip.lensModel + "\" in the file";
        return nullptr;
    }
    return best;
}

// The calibration point nearest to the clip's focal length, aperture and focus distance.
const Entry* pick_entry(const Lens& l, const DngInfo& clip, bool vignette) {
    const Entry* best = nullptr;
    double bestD = 1e30;
    auto apart = [](double a, double b, double weight) { return a > 0 && b > 0 ? weight * std::fabs(std::log(a / b)) : 0.0; };
    for (const auto& e : l.entries) {
        if (vignette ? !e.hasVignette : !e.hasWarp) continue;
        const double d = apart(e.focal, clip.focalMm, 8) + (vignette ? apart(e.aperture, clip.fNumber, 2) : 0.0) +
                         apart(std::min(e.distance, 100.0), std::min(clip.focusM, 100.0), 0.5);
        if (d < bestD) { bestD = d; best = &e; }
    }
    return best;
}

std::string describe(const Profile& p, const Lens& l, const Entry& e, bool vignette) {
    char buf[160];
    int n = std::snprintf(buf, sizeof buf, "%s", p.kind.c_str());
    if (e.focal > 0) n += std::snprintf(buf + n, sizeof buf - n, ", %.0f mm", e.focal);
    if (vignette && e.aperture > 0) n += std::snprintf(buf + n, sizeof buf - n, " f/%.1f", e.aperture);
    (void)n;
    return p.lenses.size() > 1 ? std::string(buf) + ", " + l.name : std::string(buf);
}

// ---- the Adobe profiles installed on this machine ----

struct Installed { std::string path; std::vector<std::string> name; bool preferred; };

const std::vector<Installed>& installed_profiles() {
    static std::once_flag once;
    static std::vector<Installed> list;
    std::call_once(once, [] {
        namespace fs = std::filesystem;
        std::vector<std::string> roots;
        const std::string tail = "/Adobe/CameraRaw/LensProfiles/1.0";
        for (const char* v : {"SFP_ADOBE_PROFILES", "ProgramData", "APPDATA"}) {
            const std::string dir = os::env(v);
            if (!dir.empty()) roots.push_back(std::string(v) == "SFP_ADOBE_PROFILES" ? dir : dir + tail);
        }
        roots.push_back("/Library/Application Support" + tail);
        if (!os::env("HOME").empty()) roots.push_back(os::env("HOME") + "/Library/Application Support" + tail);
        for (const auto& root : roots) {
            std::error_code ec;
            for (fs::recursive_directory_iterator it(fs::u8path(root), fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec)) {
                const fs::path& f = it->path();
                if (lower(f.extension().u8string()) != ".lcp") continue;
                // "SONY (Viltrox 28mm F4.5 FE) - RAW.lcp": the lens is named between the brackets.
                const std::string file = f.filename().u8string();
                const size_t a = file.find('('), b = file.rfind(')');
                if (a == std::string::npos || b == std::string::npos || b < a) continue;
                Installed x;
                x.path = f.u8string();
                x.name = words(file.substr(a + 1, b - a - 1));
                x.name.erase(std::remove_if(x.name.begin(), x.name.end(), [](const std::string& w) { return w == "distortion" || w == "vignette" || w == "distortion_vignette"; }),
                             x.name.end());
                // Profiles made on smaller-sensor systems do not reach a full-frame corner (and name another version of the lens).
                const std::string camera = lower(file.substr(0, a));
                if (camera.find("fujifilm") != std::string::npos || camera.find("olympus") != std::string::npos || camera.find("om digital") != std::string::npos) continue;
                x.preferred = std::find(x.name.begin(), x.name.end(), "mount") != x.name.end() && std::find(x.name.begin(), x.name.end(), "l") != x.name.end();
                if (!x.name.empty()) list.push_back(std::move(x));
            }
        }
    });
    return list;
}

}  // namespace

std::string adobe_profile_for(const DngInfo& clip, bool vignette) {
    if (clip.lensModel.empty() || os::env("SFP_ADOBE_PROFILES") == "off") return std::string();
    static std::mutex m;
    static std::map<std::string, std::string> known;
    char focalKey[32];
    std::snprintf(focalKey, sizeof focalKey, "|%.0f|%d", clip.focalMm, vignette ? 1 : 0);
    const std::string key = clip.lensModel + focalKey;
    {
        std::lock_guard<std::mutex> l(m);
        auto it = known.find(key);
        if (it != known.end()) return it->second;
    }
    const auto want = words(clip.lensModel);
    // An aperture in a lens name: "2.8", "3.5-5.6", or the "2" of "F2".
    auto aperture = [](const std::string& w) {
        return !w.empty() && std::isdigit(static_cast<unsigned char>(w[0])) && w.find("mm") == std::string::npos && (w.find('.') != std::string::npos || w.size() <= 2);
    };
    // "019" of "Contemporary 019" is the "c019" of the profile's name.
    auto same = [](const std::string& have, const std::string& w) {
        if (have == w) return true;
        const bool digits = w.size() >= 3 && std::all_of(w.begin(), w.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
        return digits && have.size() == w.size() + 1 && have.compare(1, w.size(), w) == 0;
    };
    std::vector<std::pair<double, const Installed*>> fits;
    for (const auto& x : installed_profiles()) {
        // Every aperture of the clip's lens name must be in the profile's name; then the share of common words counts.
        bool apertures = true, focalWord = false;
        int common = 0;
        for (const auto& w : want) {
            const bool in = std::any_of(x.name.begin(), x.name.end(), [&](const std::string& h) { return same(h, w); });
            common += in;
            if (!in && aperture(w)) apertures = false;
            if (in && w.find("mm") != std::string::npos) focalWord = true;
        }
        if (!apertures || (!(clip.focalMm > 0) && !focalWord)) continue;
        const double score = common / std::sqrt(static_cast<double>(want.size() * x.name.size()));
        if (score >= 0.5) fits.push_back({score + (x.preferred ? 0.05 : 0.0), &x});
    }
    std::stable_sort(fits.begin(), fits.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    // The best fit that holds the wanted model at the clip's focal length, made with a full-frame camera.
    std::string best;
    double bestScore = 0;
    for (size_t i = 0; i < fits.size() && i < 16; ++i) {
        const auto p = load(fits[i].second->path);
        if (!p->error.empty() || p->lenses.empty()) continue;
        const Lens& l = p->lenses[0];
        bool covers = false;
        for (const auto& e : l.entries)
            if ((vignette ? e.hasVignette : e.hasWarp) && (!(clip.focalMm > 0) || !(e.focal > 0) || std::fabs(std::log(e.focal / clip.focalMm)) < 0.08)) covers = true;
        if (!covers) continue;
        const double score = fits[i].first - (l.formatFactor > 1.1 ? 0.3 : 0.0);
        if (score > bestScore) { bestScore = score; best = fits[i].second->path; }
    }
    if (bestScore < 0.5) best.clear();
    std::lock_guard<std::mutex> l(m);
    known[key] = best;
    return best;
}

// The lens named in an installed profile's file name.
static std::string installed_name(const std::string& path) {
    const std::string file = file_name(path);
    const size_t a = file.find('('), b = file.rfind(')');
    return a != std::string::npos && b != std::string::npos && b > a ? file.substr(a + 1, b - a - 1) : file;
}

bool lens_vignette(const std::string& userFile, const DngInfo& clip, GainMap& out, bool& ownColour, std::string& note) {
    ownColour = true;
    std::string why;
    if (!userFile.empty()) {
        if (profile_vignette(userFile, clip, out, note)) { note = "file, " + note; return true; }
        why = "[" + note + "] ";
    }
    if (clip.shading.valid() && clip.shading.brightness() > 1.01f) {
        out = clip.shading;
        ownColour = false;   // the frame's map as it is: brightness and colour
        char buf[64];
        std::snprintf(buf, sizeof buf, "from the camera (corners x%.2f)", out.brightness());
        note = why + buf;
        return true;
    }
    const std::string adobe = adobe_profile_for(clip, true);
    if (!adobe.empty() && profile_vignette(adobe, clip, out, note)) {
        note = why + "Adobe list, " + installed_name(adobe) + note.substr(note.find(','));
        return true;
    }
    out = GainMap{};
    note = why + (clip.shading.valid() ? "none (camera Vignetting off, no Adobe profile for this lens): colour shading only" : "none");
    return false;
}

bool lens_distortion(const std::string& userFile, const DngInfo& clip, LensWarp& out, std::string& note) {
    std::string why;
    if (!userFile.empty()) {
        if (profile_distortion(userFile, clip, out, note)) { note = "file, " + note; return true; }
        why = "[" + note + "] ";
    }
    if (clip.warp.valid) { out = clip.warp; note = why + "from the camera"; return true; }
    const std::string adobe = adobe_profile_for(clip, false);
    if (!adobe.empty() && profile_distortion(adobe, clip, out, note)) {
        note = why + "Adobe list, " + installed_name(adobe) + note.substr(note.find(','));
        return true;
    }
    out = LensWarp{};
    note = why + "none";
    return false;
}

bool profile_vignette(const std::string& file, const DngInfo& clip, GainMap& out, std::string& note) {
    const auto p = load(file);
    if (!p->error.empty()) { note = file_name(file) + ": " + p->error; return false; }
    if (p->kind == "DNG") {
        const GainMap& g = p->dng.shading;
        if (!g.valid() || !(g.brightness() > 1.01f)) { note = file_name(file) + ": no vignette in this DNG (shot with the camera's Vignetting off?)"; return false; }
        // Its brightness plane alone: the colour shading stays the clip's own.
        out = GainMap{};
        out.rows = g.rows; out.cols = g.cols; out.planes = 1;
        out.areaW = g.areaW; out.areaH = g.areaH;
        out.gain.resize(static_cast<size_t>(g.rows) * g.cols);
        for (size_t i = 0; i < out.gain.size(); ++i) out.gain[i] = g.gain[i * g.planes + (g.planes == 3 ? 1 : 0)];
        char buf[64];
        std::snprintf(buf, sizeof buf, "DNG, corners x%.2f", out.brightness());
        note = buf;
        return true;
    }
    std::string why;
    const Lens* l = pick_lens(*p, clip, why);
    const Entry* e = l ? pick_entry(*l, clip, true) : nullptr;
    if (!e) { note = file_name(file) + ": " + (l ? "no vignette data for this lens" : why); return false; }
    // The falloff over the whole sensor, as a brightness map like the camera's own.
    out = GainMap{};
    out.rows = 41; out.cols = 61; out.planes = 1;
    out.areaW = static_cast<int>(kSensorW); out.areaH = static_cast<int>(kSensorH);
    out.gain.resize(static_cast<size_t>(out.rows) * out.cols);
    for (int i = 0; i < out.rows; ++i)
        for (int j = 0; j < out.cols; ++j) {
            const double x = (static_cast<double>(j) / (out.cols - 1) - 0.5) * kSensorW * kPitchMm;
            const double y = (static_cast<double>(i) / (out.rows - 1) - 0.5) * kSensorH * kPitchMm;
            const double r2 = (x * x + y * y) / (e->vignetteRadiusMm * e->vignetteRadiusMm);
            const double fall = 1 + r2 * (e->v[0] + r2 * (e->v[1] + r2 * e->v[2]));
            out.gain[static_cast<size_t>(i) * out.cols + j] = static_cast<float>(1.0 / std::clamp(fall, 0.1, 4.0));
        }
    note = describe(*p, *l, *e, true);
    return true;
}

bool profile_distortion(const std::string& file, const DngInfo& clip, LensWarp& out, std::string& note) {
    const auto p = load(file);
    if (!p->error.empty()) { note = file_name(file) + ": " + p->error; return false; }
    if (p->kind == "DNG") {
        if (!p->dng.warp.valid) { note = file_name(file) + ": no distortion profile in this DNG"; return false; }
        out = p->dng.warp;
        note = "DNG";
        return true;
    }
    std::string why;
    const Lens* l = pick_lens(*p, clip, why);
    const Entry* e = l ? pick_entry(*l, clip, false) : nullptr;
    if (!e) { note = file_name(file) + ": " + (l ? "no distortion data for this lens" : why); return false; }
    out = LensWarp{};
    out.valid = true;
    for (int i = 0; i < 7; ++i) out.c[i] = e->c[i];
    out.radius = e->warpRadiusMm / kPitchMm;
    note = describe(*p, *l, *e, false);
    return true;
}

}  // namespace sfp
