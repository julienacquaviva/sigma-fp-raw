"""Synthetic CinemaDNG frames for the geometry tests: an uncompressed 16-bit Bayer (RGGB) raster
of any size with a sharp test chart (grid lines one pixel wide, a diagonal, an off-centre
marker), the tags the plug-in's reader needs, and an active area that matches the R124
recorded sizes. No camera file is needed, so the Transform tests run without the SSD.

usage (as a module): write_dng(path, active_w, active_h) -> path"""
import struct
from pathlib import Path

import numpy as np
import tifffile


def chart(w, h):
    """Linear scene values per pixel: a 32-px checkerboard (0.06 / 0.14), dark grid lines 3 px wide
    every 64 px, a bright marker square 100 px right of and 60 px above the centre, a bright
    diagonal, a slight left-to-right ramp (so that mirror images differ) and fine texture."""
    y, x = np.mgrid[0:h, 0:w]
    v = np.where(((x // 32) + (y // 32)) % 2 == 0, 0.06, 0.14)
    v[(x % 64) < 3] = 0.01
    v[(y % 64) < 3] = 0.01
    cx, cy = w // 2, h // 2
    v[(abs(x - (cx + 100)) < 20) & (abs(y - (cy - 60)) < 20)] = 0.5
    v[np.abs((x - cx) - 2 * (y - cy)) < 2] = 0.4
    v = v + 0.05 * (x / w) + 0.02 * np.random.default_rng(1).random((h, w))
    return v


def write_dng(path, active_w, active_h, border=(8, 6), scene=None):
    """Raster = active area plus a border (as the camera's DNGs have); DefaultCrop = the active size."""
    bx, by = border
    W, H = active_w + 2 * bx, active_h + 2 * by
    black, white = 256, 4095
    full = np.zeros((H, W))
    full[by:by + active_h, bx:bx + active_w] = chart(active_w, active_h) if scene is None else scene
    raw = (black + full * (white - black)).astype('>u2')     # the reader takes 16-bit samples big-endian
    tags = [
        (254, 'I', 1, 0, True),                    # NewSubFileType: main image
        (33421, 'H', 2, (2, 2), True),             # CFARepeatPatternDim
        (33422, 'B', 4, (0, 1, 1, 2), True),       # CFAPattern RGGB
        (50706, 'B', 4, (1, 4, 0, 0), True),       # DNGVersion
        (50708, 's', 0, 'SIGMA fp', True),         # UniqueCameraModel
        (50714, 'H', 1, black, True),              # BlackLevel
        (50717, 'H', 1, white, True),              # WhiteLevel
        (50719, 'I', 2, (0, 0), True),             # DefaultCropOrigin (relative to the active area)
        (50720, 'I', 2, (active_w, active_h), True),
        (50721, '2i', 9, (10000, 10000, 0, 10000, 0, 10000, 10000, 0, 0, 10000, 0, 10000, 0, 10000, 0, 10000, 10000, 10000), True),
        (50728, '2I', 3, (1, 1, 1, 1, 1, 1), True),   # AsShotNeutral
        (50730, '2i', 1, (0, 1), True),            # BaselineExposure 0
        (50778, 'H', 1, 21, True),                 # CalibrationIlluminant1 D65
        (50829, 'I', 4, (by, bx, by + active_h, bx + active_w), True),   # ActiveArea top, left, bottom, right
    ]
    tifffile.imwrite(path, raw, byteorder='>', photometric=32803, compression=None, rowsperstrip=H, extratags=tags, metadata=None)
    return Path(path)
