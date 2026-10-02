## Downloads

| System | File | How to install |
|---|---|---|
| Windows 10 / 11 (64-bit) | `SigmaFpRaw-…-Windows-Setup.exe` | Run it. Windows SmartScreen may warn about an unknown publisher: More info > Run anyway. |
| macOS 11 or newer (Apple Silicon and Intel) | `SigmaFpRaw-…-macOS.pkg` | Right-click > Open (the package is not notarised by Apple), then follow the installer. |
| Linux (64-bit) | `SigmaFpRaw-…-all-platforms.zip` | Unzip, then `sh install-linux.sh`. |
| Any system, by hand | `SigmaFpRaw-…-all-platforms.zip` | Unzip and copy the `SigmaFpRaw.ofx.bundle` folder into the OFX plug-in folder (see `INSTALL.txt`). |

Close DaVinci Resolve before installing. Afterwards: Color page > Effects > Sigma fp > **Sigma fp RAW**, on the first node of a Sigma fp CinemaDNG clip.

## New in 1.4.0

- Runs on macOS (Apple Silicon and Intel) and Linux as well as Windows.
- Runs without an NVIDIA card: frames are developed on the processor when there is no CUDA driver (Macs, AMD and Intel graphics, NVIDIA cards older than the GTX 10 series). The picture is the same as on the GPU; it is slower.
- NVIDIA support widened from the RTX / GTX 16 series to the GTX 10 series and newer.
- Installers for Windows and macOS, and one folder that can be copied by hand on any system.

## Status

- Windows with an NVIDIA GPU: used in DaVinci Resolve Studio; real time.
- Processor path: same picture as the GPU path in the automated tests on Windows.
- macOS and Linux: the plug-in builds, loads and passes its self-test on GitHub's Mac and Linux machines. It has **not yet been tried inside DaVinci Resolve** on those systems. Reports are welcome in the issue tracker.
