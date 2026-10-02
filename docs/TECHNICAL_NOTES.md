# Sigma fp RAW: technical notes

Measurements and design notes behind the plug-in. For installing and using it, see the [README](../README.md).

The plug-in re-reads the clip's original DNG files, develops them (in Resolve's own CUDA context on an NVIDIA GPU, on the processor otherwise) and replaces its input image.

- **Real-time:** 3K 50p (A001_065, R52 mixed frames) plays at a sustained 50 fps into a UHD image on an
  RTX 4080 Laptop GPU reading from the fp SSD. Median 8 ms and p95 9 ms per frame, 995 of 1000 frames from read-ahead.
  Run `build\sfp_cli.exe play FIRST.DNG 1000 fps=50` to measure.
- **Compressed frames:** its own LJ92 decoder handles the R52 in-camera tiles (256×256, 2 components, predictor 1,
  zero-padded tiles) and uncompressed frames. Both are bit-identical to tifffile/imagecodecs.
- **3K row phase:** in sensor mode M98 the G2/B rows sit 0.5 px off the Bayer grid, measured on the raw data (G2−G1 +0.51 px,
  B−R +0.54 px on the monitor lines of A001_065). **Row Phase (3K)** re-spaces the row pairs before demosaicing. The recommended
  −0.125 plane pitch (the default) gives the best R/G/B registration.
- **Edge Anti-aliasing (3K):** the /2 binned readout leaves a 2-row stair-step on thin, sharp, near-horizontal lines that no demosaic
  can undo. The filter averages along such edges only:
  - the edge must be sharp, straight and within about 25° of horizontal;
  - the averaging length follows the stair-step period;
  - defocused or curved edges are left alone.

  Results on A001_065, default 80 (100 in brackets):
  - "Report history" lines: stair-step amplitude −36 to −44% (−42 to −56%).
  - Defocused statue: no visible change.
  - The first version smeared soft edges; this was fixed on user feedback.
- **Camera RAW controls:** laid out like Resolve's CinemaDNG panel.
- **Gyro stabilisation (v1.1, in-frame data v1.2):** when the clip carries the camera's gyro data, inside the DNG frames
  (camera build R112) or as a `<clip>.FPG` file next to them (R110/R111), the plugin stabilises the clip and corrects the
  rolling shutter with no setup. Without gyro data nothing changes. See "Gyro stabilisation" below.

## Project settings
1. Color page: put **Sigma fp RAW** (Effects > Sigma fp) on the **first node** of the original DNG clip. Grade after it.
2. Project settings > Camera RAW > CinemaDNG: set **Decode Quality: Quarter Res.** Resolve still decodes the clip for the
   node input, but the plug-in replaces it, so the lowest quality saves Resolve's own decode time.
3. Colour management:
   - Unmanaged YRGB timeline: Rec.709 / Rec.709 (defaults). Avoid pure Gamma 2.2/2.4: their infinite slope at black turns near-black noise into sparkles.
   - DaVinci Wide Gamut timeline: Color Space **DaVinci Wide Gamut**, Gamma **DaVinci Intermediate**.

## Controls
**Camera RAW** — Decode Quality (Full: RCD demosaic, Half: 2×2 bin), White Balance (As Shot, Daylight, Cloudy, Shade,
Tungsten, Fluorescent, Flash, Custom), Color Space (Rec.709, P3 D65, Rec.2020, DaVinci Wide Gamut, ACES AP0/AP1),
Gamma (Linear, 2.2, 2.4, Rec.709, sRGB, DaVinci Intermediate, ACEScct), Color Temp, Tint, Exposure, Sharpness,
Highlights, Shadows, Color Boost, Saturation, Midtones, Lift, Gain, Contrast.
- Highlight Recovery, Gamut Mapping, Apply Pre Tone Curve and Apply Soft Clip are no longer in the panel (v1.3). They were all
  off by default and stay off; the picture is unchanged. The parameters still exist, hidden, so a project that had one switched
  on renders as before.
- Editing Temp or Tint switches White Balance to Custom. As Shot shows the camera's white balance as Temp and Tint.
- White balance follows the DNG spec: ColorMatrix1/2 interpolated by temperature, and the DNG SDK temperature/tint mapping.
- These are our own implementations with Resolve's names, layout and ranges. The slider response is not Blackmagic's proprietary math.
- The DNG BaselineExposure (+3 EV on the fp) is applied, as Adobe does.

**Stabilisation (gyro)** — works only when the clip has gyro data:
- **Gyro** (read-only): the data in use with frames, sample rate, readout time, focal length and zoom, or why stabilisation is off.
  - In-frame data: "Loading gyro 40 % (N frames)" while the frames are read, then "On | gyro: N frames, in-frame, 2499.48 Hz, …".
  - A sidecar file: "On | A001_093_20261001.FPG: 293 frames, …"; when it stops before the clip does, "gyro data ends at frame N of M".
  - The text is refreshed when the node is created and when a control is changed, not by itself.
- **Stabilisation:** on by default.
- **Smoothness:** time constant of the camera-path smoothing in seconds, 0.5 by default. 0 follows the camera (rolling-shutter
  correction only); large values lock the shot. More smoothing needs more zoom.
- **Rolling Shutter Correction:** 0 to 1, 1 by default (each sensor row is corrected for the time it was read).
- **Rolling Shutter ms** and **Focal Length (mm):** filled in with the clip's own values (for example 24.83 and 28.0) once the
  gyro data is loaded. Type another value to override it; the Gyro line then says "(override)". 0, or the control's reset,
  gives the field back to the clip's value. A value you typed is never replaced, also not when the project is reopened.
  - The fields are filled when the node is created or a control is changed (a host does not let a plugin set values at any
    other time). For a sidecar clip, a cached clip or a short one that is at once. For a long in-frame clip read for the
    first time, they show 0 until the load has finished and you touch a control or reselect the node; the clip's values are
    used for the picture either way.
  - A manual lens has focal length 0 in the gyro data: stabilisation stays off until you type one.
- **Sync Offset (ms):** a fine trim on top of the sync the plugin works out (shown in the Gyro line, for example
  "sync 7.0 ms = readout/2 + exposure/2 (1/640 s)"); 0 by default and normally left there.
- **Range:** the part of the clip that smoothing and zoom are worked out for.
  - **Automatic** (default): the part this timeline clip uses, so after blading out a shaky section each remaining piece gets
    its own, smaller zoom. It relies on the host reporting the clip's frame range; the Gyro line shows what was received
    ("range: frames 301 to 780 of 1260 (automatic: host clip range 0 to 479, source frame = time +301)"). If the host reports
    nothing usable, or the clip is retimed, the whole clip is used and the line says so. Whether Resolve reports the trimmed
    range has not been verified; if the line keeps saying "whole clip" on a trimmed clip, use Manual.
  - **Manual:** **Range Start** and **Range End** are source frame numbers (the number in the DNG file name); End 0 = to the
    end. Each timeline piece has its own node, so each can have its own range.
  - **Whole Clip:** as before v1.3.
  - Frames outside the range keep its zoom and get as much correction as fits, so they never show a border.
- **Auto Zoom** and **Auto Zoom Limit:** zooms in by the smallest amount that hides the borders on every frame of the range, up
  to the limit (1.3 by default). Frames that would need more are stabilised less, so one jolt doesn't zoom the whole clip.
- **Zoom Mode:** **Fixed** (default) uses one zoom for the range. **Dynamic** lets the zoom follow what each part needs: a
  shaky section costs zoom only near itself. **Dynamic Zoom Smoothness** (4 s by default) is the time over which it looks
  ahead and eases; the zoom is never below what a frame needs. Shorter settings save more zoom but the slow push in and out
  becomes visible (see the numbers below).
- **Zoom:** manual zoom, multiplies the automatic one.
- **Gyro File:** leave empty; only for a `.FPG` file that isn't next to the DNGs (it then takes the place of the in-frame data).

**No longer in the panel (v1.3.2):** the Sigma fp group and the Advanced group. Their parameters still exist, hidden, at
their defaults (old projects load and render as before): Row Phase (3K) −0.125 and Edge Anti-aliasing (3K) 80, which act on
3K (3024×2010) frames only; Image Fit: Scale to Fit; First DNG File empty; Frame Mapping: Resolve Source Frame; Timeline
Anchor 0. The plugin gets the clip and the source frame from Resolve on the Color page. Where Resolve gives it no clip path
(for example on the Fusion page) there is now no control to point it at the DNGs: it shows an error and the node does nothing.

## Gyro stabilisation
- **In-frame data (camera build R112):** every DNG carries its slice of gyro samples in an "FPG2" block in the last 3072 bytes
  of the Sigma MakerNote (`FPGYRO_INFRAME.md` in the camera project). The plugin reads the block of every frame of the clip
  and builds the same track a sidecar file gives: sample rate from the mark clocks, frame table from the sample counts,
  readout time, mark delay and axis map from the blocks, raster, active area, frame rate and focal length from the DNG.
  - The frames are read on a background thread. Until it is done the clip plays unstabilised and the status says
    "Loading gyro N %"; frames already shown stay as they are until Resolve renders them again.
  - The result is kept in memory and in a cache file under `%LOCALAPPDATA%\SigmaFpRaw\gyro` (macOS: `~/Library/Caches/SigmaFpRaw/gyro`, Linux: `~/.cache/SigmaFpRaw/gyro`; keyed by clip path, frame count and
    the first and last file's time and size; about 54 MB per hour of footage, the folder is trimmed to 1 GB). Nothing is written
    next to the clip. `SFP_GYRO_CACHE_DIR` moves the folder; `off` disables it.
  - Time to read the blocks (32 threads; the drive is what limits it):

    | where | per 1000 frames | one hour (90,000 frames) |
    |---|---|---|
    | camera SSD over USB (exFAT), first read | 0.75 s | about 68 s (extrapolated) |
    | internal NVMe SSD, first read of new files | 0.46 s | 41 s (measured, header-only frames) |
    | internal NVMe SSD, files read before | 0.01 s | 0.9 s (measured) |
    | any drive, from the cache file | – | 0.3 s (measured) |

    So the first opening of a long clip takes about a minute per hour of footage, not the 15 s hoped for; after that it is
    immediate. One thread would need 14 s per 1000 frames on the camera SSD, which is why the scan uses 32.
  - A frame without a block (the camera refused it) loses nothing: its samples are in the next frame's block. Where samples
    are missing (blocks damaged or removed, a pause longer than the camera's 240 ms ring, frames of another take), the
    mark clock gives the length of the gap, the frames in it are not stabilised, and the correction fades out and in over
    one second around it. The status then says "no gyro data on K of N frames".
- **Sidecar file (R110/R111):** one `<clip>.FPG` per clip next to the DNGs (`FPGYRO_FORMAT.md`). The plugin takes the `.FPG`
  whose name starts the DNG names (`A001_092.FPG` for `A001_092_20261001_000001.DNG`), else the one named like the folder,
  else the only `.FPG` in the folder. It is used when the frames carry no blocks; with both, the frames win.
- DNG number N is gyro frame N − 1 in both cases.
- **Method:** the gyro rate is integrated into a camera orientation (quaternions). The orientation at each frame's readout
  middle (frame table and mark delay from the file) is smoothed with a zero-phase Gaussian low-pass. Each output pixel is
  turned from the smoothed camera into the real camera at the time its sensor row was read, and projected with a pinhole
  model. The warp is part of the existing develop kernel, so it costs no extra pass.
- **Frame times (v1.3.1).** The camera marks each frame when its recorder task gets it. When that task is held up the mark
  is late, by up to 37 ms on A001_002, while the sensor went on exposing at its regular rate. So a frame's time is not
  taken from its own mark but from the regular cadence fitted to all marks: the lower envelope of the marks (a mark can be
  late, never early), followed over ±2 s so that slow drift between the clocks is tracked, and raised by the clip's typical
  mark lateness (0.3 to 0.45 ms on the clips measured) so that it stands for an undisturbed mark. A frame the camera dropped
  is a lasting step of whole frame periods, not a late mark, and stays a gap. The Gyro line reports "N late frame marks
  corrected" and "N frames dropped by the camera".
- **Sync (v1.3.1).** Time from the middle of a frame's readout to its mark = readout/2 + exposure/2: the mark comes about at
  the end of the readout, and the picture belongs to the middle of the exposure. The exposure time is the DNG's EXIF
  ExposureTime (followed over the clip if it changes). The camera's stored value (readout/2 + 5 ms) was right only at 1/100 s
  and is now used only when the exposure or readout time is unknown.

  | clip | mode, shutter | stored delay | readout/2 + exposure/2 | the picture says |
  |---|---|---|---|---|
  | A001_092 | HQ 25, 1/100 s | 17.0 ms | 17.4 ms | about 18.2 ms |
  | A001_001 | MQ 50, 1/640 s | 11.2 ms | 7.0 ms | 7.3 ms |
  | A001_002 | MQ 50, 1/640 s | 11.2 ms | 7.0 ms | 5.4 to 7.5 ms (noisier) |

  Two modes and two shutter speeds only: the model fits both within about 1 ms, which is the scatter of the measurement. A
  clip at a slow shutter in a fast mode (say MQ 50 at 1/60 s: 6.2 + 8.3 = 14.5 ms) is the test that would confirm it; if
  the model is right its best Sync Offset is 0 as well.
- **Image scale:** focal length from the gyro data or the DNG, pixel pitch from the gyro data. The camera writes no pitch;
  the plugin then takes the sensor width the raster stands for: 35.9 mm (the whole sensor) in general, 32.5 mm for HQ
  (3264 raster). Both are measured: on A001_001 (MQ, 28 mm) the picture moves by 1.009 / 1.006 (x / y) of what the gyro
  predicts with 35.9 mm; on A001_092 (HQ, 28 mm) it took 32.5 mm (49.1 px per degree), so HQ reads a narrower part of the
  sensor. That is what the earlier "10.5 % unexplained" was. Other modes (UHD, XQ, LQ, S16, DC crop) are not measured yet.
- **Measured on A001_092** (HQ 3264×2170, 25 fps, 28 mm, a take with ±20° pans and tilts; frames 1–290, raster px):

  | | unstabilised | stabilised, zoom 1 | stabilised, all defaults |
  |---|---|---|---|
  | frame-to-frame shift, RMS | 58.0 | 9.7 | 27.9 |
  | change of shift between frame pairs, median | 7.8 | 0.9 | 1.6 |
  | deviation from the intended (smoothed) path, median | – | 1.3 | 1.4 |

  With all defaults this take sits at the 1.3 zoom limit and 249 of 292 frames are stabilised less than asked; no frame shows
  a border. Top-against-bottom shear per frame pair (median): 1.76 px unstabilised, 0.50 px with rolling-shutter correction.
- **Rolling shutter, checked on three clips** (A001_092, and the camera-written A001_093 and A001_094). The gyro predicts the
  shear between a top and a bottom band for every frame pair; the slope of the measured shear against the predicted one is 1
  for an uncorrected picture, 0 for a corrected one and 2 for a correction applied the wrong way round. Horizontal / vertical:

  | clip | unstabilised | amount 0 | amount 1 | amount −1 (reversed) |
  |---|---|---|---|---|
  | A001_092 | 1.08 / 1.08 | 1.19 / 1.12 | 0.13 / 0.07 | 2.13 / 2.11 |
  | A001_093 | 0.92 / 0.93 | 0.78 / 0.93 | −0.07 / −0.01 | 1.71 / 1.87 |
  | A001_094 | 0.76 / 0.91 | 0.82 / 0.93 | −0.18 / −0.14 | 1.83 / 1.89 |

  So the first raster row is read first, the file's readout time needs no factor, and amount 1 removes the shear. Smoothness does
  not change it: each row is taken from the camera's own orientation at the time that row was read.
- **End of the gyro data:** when the file has fewer frames than the clip has DNGs (or, with an unknown DNG count, its
  "stopped early" flag is set), the correction, the rolling-shutter correction and the automatic zoom fade to none over the last
  second of gyro data. The last gyro frame and every frame after it are the recorded picture (manual Zoom still applies). On
  A001_093 (293 gyro frames of 1045) the hard cut of v1.1.0 was a 289 px jump plus a 1.3× zoom step at frame 294; now there is none.
- **Unknown values in the file:** pixel pitch 0 uses the calibrated raster width above; exposure 0 needs nothing (the exposure
  time isn't used); focal length 0 keeps stabilisation off until Focal Length is set.
- **Cutting out a rough part (v1.3).** Smoothing, zoom and the zoom limit look at the used range only. At each end of a cut
  range the smoothing either stops (as at a clip end) or runs on into the frames beyond the cut; both are tried and the one
  needing less zoom is used. Stopping avoids a rough stretch just beyond the cut pulling on the path; running on avoids the
  path lagging where the camera was moving through the cut.

  | clip (0.5 s smoothness, limit 1.3) | whole clip | a piece | dynamic 2 s | dynamic 4 s |
  |---|---|---|---|---|
  | A001_093 (293 frames) | 1.300, 27 frames reduced | frames 1–120: 1.069, none reduced | 1.07–1.29, mean 1.20, 8.8 %/s | 1.22–1.30, mean 1.28, 2.0 %/s |
  | A001_094 (293) | 1.166 | frames 181–266: 1.100 | 1.10–1.17, mean 1.14, 1.9 %/s | 1.12–1.17, mean 1.15, 1.0 %/s |
  | A001_095 (1294) | 1.300, 661 reduced | frames 1–125: 1.149, none reduced | 1.09–1.30, mean 1.25, 6.6 %/s | 1.13–1.30, mean 1.28, 2.1 %/s |
  | A001_096 (1261) | 1.300, 673 reduced | frames 246–370: 1.058, none reduced | 1.09–1.30, mean 1.27, 10.0 %/s | 1.29–1.30, mean 1.30, 0.5 %/s |

  "%/s" is the fastest zoom change. The "piece" column was measured with the smoothing stopping at the cut; with the choice
  described above it can only be the same or lower. These takes are rough throughout (half of the frames of A001_095 and 096 are over the
  limit), so the dynamic zoom gains little on them and at 2 s it moves fast enough to be seen; cutting to a calm piece is
  what brings the zoom down. On a calm take with one rough spot (the synthetic test clip: 1.04 needed, 1.38 for two rough
  seconds) the dynamic zoom averages 1.14 at 2 s and 1.23 at 4 s against 1.38 fixed.
- **Not covered:** lens distortion (pinhole only), the roll sign (the picture measurement agrees with +Z, but with a
  correlation of only 0.8 to 0.9), other resolutions and DC crop (pitch fallback untested), and the status label and parameter
  layout inside Resolve (tested in the independent host; both render paths, host memory and GPU image, give the same picture).
  No camera clip with in-frame blocks exists yet: that path is tested on frames with synthesised blocks and on blocks written
  by the R112 firmware code in emulation.

## Limits
- GPU development needs an NVIDIA card (GeForce GTX 10 series or newer) with CUDA. If Resolve is in OpenCL mode, the plug-in runs in its own CUDA context and copies the result through host memory, which is slower.
- Without a CUDA driver (macOS, AMD and Intel graphics) or on an older NVIDIA card, frames are developed on the processor: same picture, slower. `SFP_CPU=1` forces this path.
- The processor path against the GPU path on three real frames (16-bit output): mean difference 0.02 of 65535, largest 85.
- Lens opcodes (OpcodeList3) are not applied. Noise reduction: none (use Resolve's).
- Retimes and compound clips aren't verified; frames map one per source frame.
- Edge Anti-aliasing reduces the 3K stair-steps; it can't restore detail the binned readout never captured.

## Build and tests
- `python build.py` builds the OFX bundle and `build/sfp_cli` for this machine; `--targets all` cross-compiles Windows, macOS and Linux with zig; `--package` writes the zip. `tools/make_ptx.py` compiles the CUDA kernels to PTX with NVRTC (from Resolve's Windows install) into `src/generated`.
- `sfp_cli selftest [FILE.FPG]` develops a synthetic frame and checks the picture; it needs no footage and runs on every system (used by the GitHub build).
- `python tools/check_plugin.py` loads the built plug-in library as a host does.
- `python tests/test_ofx.py [CLIP_DIR]` loads the built `.ofx` in an independent OFX host and runs 13 cases against real frames:
  source path and frame mapping, compressed and uncompressed frames equal to the CLI, strides and windows, render scale,
  errors, anchor mode, the hidden controls, First DNG File alone, exposure, UHD fit.
- `python tests/test_gyro.py [CLIP_DIR]` tests the stabilisation (40 cases, with the used range, the dynamic zoom, the
  prefilled fields, the regular-cadence frame times and the sync model; the MQ 50 clips are looked for in `E:/CINEMA`): the parser on `tests/data/A001_092.FPG`, synthetic gyro
  files against a numpy model (known rotation → shift, rolling shutter, sync, zoom, fade-out, file lookup), both render paths, the
  GPU warp in the OFX host, and the numbers above on the real clips (default `F:/CINEMA/A001_092`, with `A001_093` and `A001_094`
  beside it; read-only; skipped if not connected).
- `python tests/test_inframe.py [CLIP_DIR]` tests the in-frame gyro path (18 cases): the block parser, the track against the
  `.FPG` track, missing, damaged, capped and foreign blocks, mixed clips, the cache file, the background load and its status
  text, blocks written by the R112 firmware in emulation (needs the camera project and `unicorn`), the same picture through
  both paths on real frames (default `F:/CINEMA/A001_095`), and the scan time. `tests/make_inframe_clip.py CLIP_DIR OUT_DIR`
  writes copies of a clip's frames with blocks made from its `.FPG`.
- `sfp_cli develop IN.DNG OUT.tif [phase=… aa=… exposure=… gamma=… cs=…]` develops one frame to a 16-bit TIFF.
- `sfp_cli gyro FILE.FPG [frame=N] [smooth= rs= sync= focal= readout= autozoom= maxzoom= zoom= range=A-B zoommode=dynamic zoomsmooth=] [pt=x,y …]` prints a gyro file's header,
  status and one frame's warp as JSON; given a clip's DNG it reports the gyro data the plugin uses for it. `sfp_cli fpg2 FRAME.DNG`
  prints one frame's block; `sfp_cli fpg2scan FIRST.DNG` times reading the blocks of a clip.
