## What's new

- **Fix: blown highlights came out yellow with a preset or Custom white balance** (Develop RAW on). They are neutral again, as they already were with As Shot. Colours below clipping are unchanged.

## Downloads

| System | File | How to install |
|---|---|---|
| Windows 10 / 11 (64-bit) | `SigmaFpRaw-…-Windows-Setup.exe` | Run it. Windows SmartScreen may warn about an unknown publisher: More info > Run anyway. |
| macOS 11 or newer (Apple Silicon and Intel) | `SigmaFpRaw-…-macOS.pkg` | Right-click > Open (the package is not notarised by Apple), then follow the installer. |
| Linux (64-bit) | `SigmaFpRaw-…-all-platforms.zip` | Unzip, then `sh install-linux.sh`. |
| Any system, by hand | `SigmaFpRaw-…-all-platforms.zip` | Unzip and copy the `SigmaFpRaw.ofx.bundle` folder into the OFX plug-in folder (see `INSTALL.txt`). |

Close DaVinci Resolve before installing. Afterwards: Color page > Effects > Sigma fp > **Sigma fp RAW**, on the first node of a Sigma fp CinemaDNG clip.
