#!/bin/sh
# Copies SigmaFpRaw.ofx.bundle into the OFX plug-in folder DaVinci Resolve loads.
cd "$(dirname "$0")" || exit 1
SRC="SigmaFpRaw.ofx.bundle"
DEST="/usr/OFX/Plugins"

if [ ! -f "$SRC/Contents/Linux-x86-64/SigmaFpRaw.ofx" ]; then
    echo "$SRC was not found next to this script."
    echo "Unzip the whole download first, then run this script from the unzipped folder."
    exit 1
fi

SUDO=""
[ "$(id -u)" -eq 0 ] || SUDO="sudo"
echo "Installing Sigma fp RAW into $DEST"
$SUDO mkdir -p "$DEST" || exit 1
$SUDO rm -rf "$DEST/$SRC"
$SUDO cp -R "$SRC" "$DEST/" || { echo "The copy failed."; exit 1; }
$SUDO chmod -R a+rX "$DEST/$SRC"

echo
echo "Sigma fp RAW is installed in $DEST/$SRC"
echo "Start DaVinci Resolve: Color page > Effects > Sigma fp > Sigma fp RAW."
