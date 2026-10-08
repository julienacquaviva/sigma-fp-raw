## What's new (pre-release, for testing)

- **Fix: Vignette Correction with Develop RAW on put faint green and magenta patches into dim areas.**
- **"Sigma fp RAW could not be run successfully": the plug-in now says why.** The reason is written to `SigmaFpRaw/log.txt` in the user's local app-data folder (`%LOCALAPPDATA%` on Windows, `~/Library/Caches` on a Mac), and shown in the Clip line of Clip Info after any control is changed.
- With Develop RAW off the plug-in uses much less GPU memory.
- Lens Correction is laid out as two blocks: Vignette Correction with Resolve Gamma, Vignette Status and Vignette Profile; Distortion Correction with Distortion Status and Distortion Profile. Resolve Gamma is Off until it is needed and then starts at Rec.709.
- Stabilisation: the tick comes first, Gyro Status underneath; Rolling Shutter Correction is now Rolling Shutter Fix; the zoom Smoothness starts at 3.

## Downloads

| System | File | How to install |
|---|---|---|
| Windows 10 / 11 (64-bit) | `SigmaFpRaw-…-Windows-Setup.exe` | Run it. Windows SmartScreen may warn about an unknown publisher: More info > Run anyway. |
| macOS 11 or newer (Apple Silicon and Intel) | `SigmaFpRaw-…-macOS.pkg` | Right-click > Open (the package is not notarised by Apple), then follow the installer. |
| Linux (64-bit) | `SigmaFpRaw-…-all-platforms.zip` | Unzip, then `sh install-linux.sh`. |
| Any system, by hand | `SigmaFpRaw-…-all-platforms.zip` | Unzip and copy the `SigmaFpRaw.ofx.bundle` folder into the OFX plug-in folder (see `INSTALL.txt`). |

Close DaVinci Resolve before installing. Afterwards: Color page > Effects > Sigma fp > **Sigma fp RAW**, on the first node of a Sigma fp CinemaDNG clip.
