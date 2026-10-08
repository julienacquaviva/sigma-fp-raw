# Sigma fp RAW: changes by version

## New in 1.9.6

- Develop RAW off: the host's frame is the timeline's frame with the clip placed in it by the host's input scaling. The
  placement is now a choice, Resolve Input Scaling (Scale to Fit / Fill); with nothing to move (no stabilisation, no lens
  correction, Transform and Fit at their defaults) the host's picture is copied as it is. 1.9.5 took the whole frame as the
  clip's frame, which put the stabilisation and the lens corrections out of place beside bars; 1.9.4 and earlier assumed
  Scale to Fit and blacked out the sides of a clip that Resolve had scaled to fill.

## New in 1.9.5

- Develop RAW off: the host's picture is taken as the clip's whole frame, whatever shape the DNG's crop tags give (a clip
  whose tags said 16:9 while Resolve showed it 2:1 was cut to 16:9 with black bars).

- The node's frame is asked to be the timeline frame only with Fit Width / Fit Height; with Scale to Fit the host's own
  frame stands (a 2:1 clip in a 16:9 timeline was reported cropped with the default settings).

## New in 1.9.4

- Mac: the GPU develops the picture, through Metal (Apple Silicon and Intel). The kernels are compiled on the Mac at the
  first render; should Metal fail, the processor does the work as before. Same picture as the processor path.

## New in 1.9.3

- The groups come in the order Clip Info, Transform, Camera RAW, Lens Correction, Stabilisation.

## New in 1.9.2

- Develop RAW is off by default: a new node keeps Resolve's own picture and colour, and the plug-in adds the Transform, the
  stabilisation and the lens corrections. Tick Develop RAW for the plug-in's own development. Nodes made with 1.8 or earlier
  have no stored value for this switch and open with it off: tick it on those nodes to get their picture back.

## New in 1.9.1

- Vignette Correction also works with Develop RAW off: the vignette's brightness is removed from Resolve's own picture in
  linear light. Set Resolve Gamma (Camera RAW group) to the Gamma of Resolve's Camera RAW panel: Linear, Gamma 2.2 / 2.4 / 2.6,
  Rec.709, sRGB, Blackmagic Design Film (also 4K and 4.6K Film), DaVinci Intermediate or ACEScct. The lens's colour shading
  is only corrected with Develop RAW on.
- Vignette Correction and Distortion Correction are off by default.
- The groups are folded by default, except Clip Info.

## New in 1.9.0

- Develop RAW (Camera RAW group, on by default). Switched off, the picture and the colour stay Resolve's own: its Camera RAW
  settings (for example Blackmagic Design colour space and Blackmagic Design Film gamma), your colour pipeline and PowerGrades
  work as without the plug-in, and the plug-in only does the Transform, the gyro stabilisation and the distortion correction
  on that picture. It costs some sharpness (Resolve's picture at the timeline resolution is resampled once more), and the
  vignette correction, the 3K corrections and the Camera RAW controls of the plug-in do nothing in that mode. Leave Resolve's
  input scaling at Scale to Fit (its default).

## New in 1.8.1

- Lens Correction takes, for each of the two corrections, in this order: the profile file you picked; else the lens's own
  profile in the clip (out of the camera); else the Adobe lens profile installed on the computer whose name fits the
  clip's lens (found by itself). The Correction line names the source.

## New in 1.8.0

- Lens Correction: a Vignette Profile and a Distortion Profile file field. Empty = the lens's own profile from the clip.
  A file replaces it: a DNG shot with the same lens (a clip frame, or a still taken with the camera's Vignetting on Auto),
  an Adobe lens profile (.lcp) or a Lensfun database file (.xml; the lens is found by the name the clip carries). The
  calibration point nearest to the clip's focal length, aperture and focus distance is used, scaled from the profile's
  camera to the fp's sensor and to the clip's sensor window. The Correction line says what was taken, or why a file was
  not used. Fisheye profiles and Gyroflow lens files are not supported.

## New in 1.7.2

- A Lens Correction group with two switches, both on by default. They use the lens's own profile, which the camera writes
  into every frame; the Correction line in Clip Info says what the clip carries.
  - Vignette Correction: removes the vignette and the colour shading, on the raw data before everything else. The vignette
    is only in the frames when the camera's Vignetting compensation was set to Auto.
  - Distortion Correction: straightens the picture in the same single resampling as the fit, the Transform and the
    stabilisation.

## New in 1.6.1

- A Clip Info block at the top: shutter speed and angle, ISO, frame rate, aspect ratio, bit depth, sensor window, recorded
  resolution, focal length, and the crop against full frame at the current frame (sensor window, fit, Transform zoom
  and the stabiliser's zoom). It refreshes when a control is changed, not during playback.
- Stabilisation is off by default; when switched on it starts at Smoothness 0.1 s with Zoom Mode Dynamic. Existing nodes keep
  their values.
- Dynamic Zoom Smoothness is now called Smoothness and is greyed out while Zoom Mode is Fixed.
- Fit sits in the Transform group. Fit Width / Fit Height fit the timeline frame, also when the host hands the node a frame
  of the clip's own shape.
- The Gyro File field is gone from the panel (projects that set it still use it).

## New in 1.5.1

- The single resampling (fit, Transform, stabilisation) now uses a Lanczos-3 kernel instead of bilinear: sharper when
  enlarging, free of aliasing when making the picture smaller, no ringing at edges. Still real time on the GPU.

## New in 1.5.0

- A Fit choice (Scale to Fit, Fit Width, Fit Height) and a Transform group with the controls of Resolve's Transform panel:
  Zoom X / Y with Link, Position, Rotation Angle, Anchor Point, Pitch, Yaw, Flip. All keyframable.
- Fit, Transform and stabilisation are one resampling of the full-resolution developed frame: no second scaling.
- With the controls neutral the picture is exactly that of 1.4.2.

## New in 1.4.2

- Row Phase and Edge Anti-aliasing apply only to 3K frames from the 2x2 binned readout (old MQ). The 1:1 sensor windows
  of the same size (R122 / R123 MQ 50, R124 3000x2000 recordings) are left alone. The Source line says which and why.
- A frame whose compressed data is damaged no longer stops the clip: the nearest earlier frame is shown, with a warning.

## New in 1.4.1

- Gyro image scale from the camera's sensor window: R124 modes (3:2, 16:9, 2:1 with and without DC Crop, M43, S16), the
  R123 and R122 MQ 50 window, older DC Crop clips. HQ is now taken at the whole sensor width, like the other older modes.

## New in 1.4.0

- Runs on macOS (Apple Silicon and Intel) and Linux as well as Windows.
- Runs without an NVIDIA card: frames are developed on the processor when there is no CUDA driver (Macs, AMD and Intel graphics, NVIDIA cards older than the GTX 10 series). The picture is the same as on the GPU; it is slower.
- NVIDIA support widened from the RTX / GTX 16 series to the GTX 10 series and newer.
- Installers for Windows and macOS, and one folder that can be copied by hand on any system.
