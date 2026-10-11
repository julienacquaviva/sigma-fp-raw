## What's new

- **Camera Matrix: Chart-Fitted (daylight)**, a new choice in Camera RAW (Develop RAW on): a ColorChecker fit of the fp's sensor in place of the file's own matrices, with the same white balance. *As Recorded* stays the default. Contributed by lukaDen.
- **ARRI Wide Gamut 3** as an output Color Space. Contributed by lukaDen.
- **Linux: the plug-in no longer shows "Failed to load" in Resolve.** Contributed by jedypod.
- Vignette Correction with Develop RAW on no longer puts faint green and magenta patches into dim areas.
- When Resolve says "Sigma fp RAW could not be run successfully", the reason is now written to `SigmaFpRaw/log.txt` in the user's local app-data folder (`%LOCALAPPDATA%` on Windows, `~/Library/Caches` on a Mac) and shown in the Clip line of Clip Info.
- Lens Correction and Stabilisation are laid out with their own status lines; less GPU memory is used with Develop RAW off.

## Downloads

| System | File | How to install |
|---|---|---|
| Windows 10 / 11 (64-bit) | `SigmaFpRaw-…-Windows-Setup.exe` | Run it. Windows SmartScreen may warn about an unknown publisher: More info > Run anyway. |
| macOS 11 or newer (Apple Silicon and Intel) | `SigmaFpRaw-…-macOS.pkg` | Right-click > Open (the package is not notarised by Apple), then follow the installer. |
| Linux (64-bit) | `SigmaFpRaw-…-all-platforms.zip` | Unzip, then `sh install-linux.sh`. |
| Any system, by hand | `SigmaFpRaw-…-all-platforms.zip` | Unzip and copy the `SigmaFpRaw.ofx.bundle` folder into the OFX plug-in folder (see `INSTALL.txt`). |

Close DaVinci Resolve before installing. Afterwards: Color page > Effects > Sigma fp > **Sigma fp RAW**, on the first node of a Sigma fp CinemaDNG clip.
