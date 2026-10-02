# Sigma fp RAW

**A CinemaDNG developer and gyro stabiliser for Sigma fp footage, as an OpenFX plug-in for DaVinci Resolve.**

[![build](https://github.com/julienacquaviva/sigma-fp-raw/actions/workflows/build.yml/badge.svg)](https://github.com/julienacquaviva/sigma-fp-raw/actions/workflows/build.yml)
[![release](https://img.shields.io/github/v/release/julienacquaviva/sigma-fp-raw)](https://github.com/julienacquaviva/sigma-fp-raw/releases/latest)
[![licence](https://img.shields.io/badge/licence-MIT-blue)](LICENSE)

Sigma fp RAW reads the original DNG frames of a clip itself and develops them in place of Resolve's
own decoder. You put it on the first node of the clip and grade after it.

- **Real-time on an NVIDIA GPU.** 3K 50p plays at 50 fps into a UHD timeline on an RTX 4080 laptop (about 6 to 8 ms per frame).
- **Camera RAW controls** laid out like Resolve's CinemaDNG panel: white balance, colour space, gamma, exposure, highlights, shadows, and the rest.
- **Compressed and uncompressed frames**: lossless JPEG (LJ92) tiles and packed 12-bit strips.
- **Sigma fp 3K fixes**: the row offset of the binned 3K readout is corrected, and the stair-steps it leaves on sharp near-horizontal edges are smoothed.
- **Gyro stabilisation and rolling-shutter correction** with no setup, for clips that carry gyro data.
- **Windows, macOS and Linux.** No NVIDIA card? It develops on the processor instead, with the same picture.

## Download and install

Get the files from the **[latest release](https://github.com/julienacquaviva/sigma-fp-raw/releases/latest)**. Close DaVinci Resolve first.

| System | Download | Install |
|---|---|---|
| Windows 10 / 11 | `SigmaFpRaw-…-Windows-Setup.exe` | Run it. If SmartScreen warns about an unknown publisher: **More info > Run anyway**. |
| macOS 11 or newer | `SigmaFpRaw-…-macOS.pkg` | **Right-click > Open**, then follow the installer. (The package is not notarised by Apple, so a double-click is refused.) |
| Linux | `SigmaFpRaw-…-all-platforms.zip` | Unzip, then run `sh install-linux.sh`. |

### Or copy one folder by hand

`SigmaFpRaw-…-all-platforms.zip` contains a single folder, `SigmaFpRaw.ofx.bundle`, that works on all three systems.
Unzip it and copy that folder into the OFX plug-in folder:

| System | Copy `SigmaFpRaw.ofx.bundle` into |
|---|---|
| Windows | `C:\Program Files\Common Files\OFX\Plugins` |
| macOS | `/Library/OFX/Plugins` |
| Linux | `/usr/OFX/Plugins` |

Create the folder if it does not exist. On a Mac, also run this once in Terminal, because macOS holds back downloaded files:

```bash
sudo xattr -dr com.apple.quarantine /Library/OFX/Plugins/SigmaFpRaw.ofx.bundle
```

The zip also has `Install-Windows.bat`, `Install-macOS.command` and `install-linux.sh`, which do the same copy for you.

To uninstall, delete `SigmaFpRaw.ofx.bundle` from that folder (on Windows the installer also adds an entry under *Installed apps*).

## Compatibility

| System | Graphics | How frames are developed |
|---|---|---|
| Windows 10 / 11, 64-bit Intel or AMD processor | NVIDIA GeForce GTX 10 series or newer | GPU (CUDA), real time |
| Windows 10 / 11, 64-bit Intel or AMD processor | AMD, Intel, older NVIDIA | Processor |
| macOS 11 or newer, Apple Silicon or Intel | any | Processor |
| Linux, 64-bit Intel or AMD processor (glibc 2.17 or newer) | NVIDIA GeForce GTX 10 series or newer | GPU (CUDA) |
| Linux, 64-bit Intel or AMD processor | AMD, Intel | Processor |

- Works in DaVinci Resolve and DaVinci Resolve Studio. Developed with Resolve 21 on Windows.
- Intel and AMD processors need AVX2 (Intel from 2013, AMD from 2015).
- The processor path gives the same picture as the GPU path but is not real time: a UHD frame takes roughly
  0.15 s on a 24-core laptop processor. **Decode Quality: Half Res.** is faster.
- Not supported yet: Windows on ARM, and GPU development on AMD, Intel and Apple graphics (OpenCL, Metal).

**What has been tested.** Windows with an NVIDIA GPU is used in Resolve day to day. The processor path is checked against
the GPU path by the test suite on Windows. The macOS and Linux builds are built, loaded and self-tested on GitHub's
machines for every change, but they have not yet been tried inside DaVinci Resolve. If you try one, please
[open an issue](https://github.com/julienacquaviva/sigma-fp-raw/issues) and say how it went.

## Use it in Resolve

1. **Color page**: open *Effects*, find **Sigma fp RAW** in the *Sigma fp* group, and drop it on the **first node** of a Sigma fp CinemaDNG clip. Grade on the nodes after it.
2. **Project Settings > Camera RAW > CinemaDNG**: set *Decode Quality* to **Quarter Res.** Resolve still decodes the clip for the node's input, but the plug-in replaces that picture, so the lowest quality only saves time.
3. **Colour management**:
   - Unmanaged timeline: leave the plug-in at Rec.709 / Rec.709.
   - DaVinci Wide Gamut timeline: set the plug-in to *Color Space* **DaVinci Wide Gamut** and *Gamma* **DaVinci Intermediate**.

The *Source* line at the top of the panel shows the clip, its frame size, ISO, frame rate and as-shot white balance.

## Controls

**Camera RAW**

| Control | What it does |
|---|---|
| Decode Quality | *Full Res.* (RCD demosaic) or *Half Res.* (2×2 binned, fastest) |
| White Balance | As Shot, Daylight, Cloudy, Shade, Tungsten, Fluorescent, Flash, Custom. Editing *Color Temp* or *Tint* switches to Custom. |
| Color Space | Rec.709, P3 D65, Rec.2020, DaVinci Wide Gamut, ACES AP0, ACES AP1 |
| Gamma | Linear, 2.2, 2.4, Rec.709, sRGB, DaVinci Intermediate, ACEScct |
| Exposure, Sharpness, Highlights, Shadows, Color Boost, Saturation, Midtones, Lift, Gain, Contrast | As in Resolve's CinemaDNG panel: same names, layout and ranges. The response is this plug-in's own, not Blackmagic's. |

**Stabilisation (gyro)**

| Control | What it does |
|---|---|
| Gyro | Read-only status: the gyro data in use, or why stabilisation is off |
| Stabilisation | On by default; does nothing on clips without gyro data |
| Smoothness | Seconds of camera-path smoothing. 0 corrects the rolling shutter only; large values lock the shot. |
| Rolling Shutter Correction | 0 to 1; 1 corrects every sensor row for the time it was read |
| Rolling Shutter ms, Focal Length (mm) | Filled in from the clip; type a value to override it (needed for manual lenses) |
| Sync Offset (ms) | Fine trim; normally 0 |
| Range | The part of the clip that smoothing and zoom are worked out for: Whole Clip, Automatic, or Manual frame numbers |
| Auto Zoom, Auto Zoom Limit, Zoom Mode, Zoom | Zooms in just enough to hide the borders; *Dynamic* lets the zoom follow what each part of the clip needs |

## Gyro stabilisation

Stabilisation needs gyro data recorded with the clip. A stock Sigma fp does not write any: the data comes from a
modified camera firmware, which is a separate project and not part of this repository. The plug-in reads it from
the DNG frames themselves, or from a `.FPG` file next to them. On every other clip the stabilisation group simply stays off.

How the sync, the rolling-shutter model and the zoom work, with measurements, is in
[docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md).

## Limits

- Lens corrections stored in the DNG (opcode lists) are not applied, and there is no noise reduction; use Resolve's.
- Retimed clips and compound clips are not verified; frames map one to one to source frames.
- The plug-in needs the clip's file path from Resolve, which it gets on the Color page. On the Fusion page it shows an error.
- Gyro stabilisation uses a pinhole model: lens distortion is not corrected.

## Troubleshooting

- **The effect is not in the list.** Check that `SigmaFpRaw.ofx.bundle` sits directly in the OFX plug-in folder, then restart Resolve. *Preferences > Video Plugins* lists plug-ins that failed to load.
- **macOS says the plug-in is damaged or from an unidentified developer.** Run the `xattr` command from the install section.
- **"No DNG source".** Apply the effect to the original CinemaDNG clip on the Color page, not to a render or a compound clip.
- **Playback is slow.** Without an NVIDIA card the processor does the work. Set *Decode Quality* to Half Res., or render a cache.

## Build from source

You need Python 3 and [zig](https://ziglang.org) 0.14 (`pip install "ziglang>=0.14,<0.15"`). One machine can build every target.

```bash
python build.py --targets all --package
```

This writes `dist/SigmaFpRaw.ofx.bundle` and `dist/SigmaFpRaw-<version>-all-platforms.zip`. Without options it
builds for the machine you are on. The Windows installer is made with Inno Setup from `packaging/windows-installer.iss`,
the macOS package with `pkgbuild`; the [GitHub workflow](.github/workflows/build.yml) does both and publishes a release for every `v*` tag.

```bash
build/sfp_cli selftest tests/data/A001_092.FPG
```

The self-test develops a synthetic frame and needs no footage. `tests/` holds the full suites, which run on Windows
against real clips (see the technical notes).

The CUDA kernels in `src/kernels.cu` are compiled to PTX ahead of time (`tools/make_ptx.py`, using the NVRTC library that
ships with Resolve on Windows) and kept in `src/generated`, so a build needs no NVIDIA software. The same file is
compiled as plain C++ for the processor path.

## Licence

[MIT](LICENSE). The OpenFX API headers in `third_party/openfx` are BSD 3-Clause.

This project is not affiliated with Sigma or Blackmagic Design.
