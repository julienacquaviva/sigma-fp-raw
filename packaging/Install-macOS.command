#!/bin/bash
# Copies SigmaFpRaw.ofx.bundle into the OFX plug-in folder DaVinci Resolve loads.
cd "$(dirname "$0")" || exit 1
SRC="SigmaFpRaw.ofx.bundle"
DEST="/Library/OFX/Plugins"

if [ ! -f "$SRC/Contents/MacOS/SigmaFpRaw.ofx" ]; then
    echo "$SRC was not found next to this script."
    echo "Unzip the whole download first, then run this script from the unzipped folder."
    exit 1
fi
if pgrep -x "Resolve" >/dev/null; then
    echo "DaVinci Resolve is running. Quit it, then run this script again."
    exit 1
fi

echo "Installing Sigma fp RAW into $DEST (your Mac password is needed for this folder)."
sudo mkdir -p "$DEST" || exit 1
sudo rm -rf "$DEST/$SRC"
sudo cp -R "$SRC" "$DEST/" || { echo "The copy failed."; exit 1; }
# Downloaded files are quarantined by Gatekeeper; the plug-in is signed on this Mac instead.
sudo xattr -dr com.apple.quarantine "$DEST/$SRC" 2>/dev/null
sudo codesign --force --sign - "$DEST/$SRC/Contents/MacOS/SigmaFpRaw.ofx" 2>/dev/null

echo
echo "Sigma fp RAW is installed in $DEST/$SRC"
echo "Start DaVinci Resolve: Color page > Effects > Sigma fp > Sigma fp RAW."
