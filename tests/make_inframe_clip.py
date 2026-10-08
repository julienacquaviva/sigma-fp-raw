"""Synthesise CinemaDNG frames with in-frame gyro blocks ("FPG2", FPGYRO_INFRAME.md as built in
R112) from a clip and its .FPG sidecar, the way the camera's in-frame build writes them. No
camera clip with these blocks exists yet; this stands in for one. The block layout and the
slice rules are checked against the firmware itself in tests/test_inframe.py (the firmware's
frame hook run in emulation by the camera project's verify_gyro.py).

Each output frame is a copy of a source frame (or only its header, `--headers-only`: enough for
the gyro loader, not for developing the picture) with the block written into the last 3072 bytes
of the Sigma MakerNote, found through IFD0 -> EXIF IFD (0x8769) -> MakerNote (0x927C). Like the
firmware, nothing is written unless that region is all zero.

usage: python tests/make_inframe_clip.py CLIP_DIR OUT_DIR [--fpg FILE] [--frames N] [--headers-only]
The source clip is only read. OUT_DIR is created; put it under tests/scratch/."""
import argparse, struct
from pathlib import Path

import numpy as np

REGION = 0xC00
CAP = 500
RING = 600
PREROLL = 500


def read_fpg(path):
    d = Path(path).read_bytes()
    assert d[:4] == b'FPGY'
    fc, sc, rate_mhz = struct.unpack_from('<III', d, 8)
    h = dict(frames=fc, samples=sc, rate=rate_mhz / 1000.0, lsb=struct.unpack_from('<f', d, 20)[0],
             readout_us=struct.unpack_from('<I', d, 44)[0], mark_delay_us=struct.unpack_from('<i', d, 56)[0],
             axes=struct.unpack_from('<3b', d, 60), flags=d[63],
             resolution=struct.unpack_from('<I', d, 72)[0], dc_crop=struct.unpack_from('<I', d, 76)[0],
             bit_depth=struct.unpack_from('<I', d, 80)[0], sensor_mode=struct.unpack_from('<I', d, 84)[0])
    table = np.frombuffer(d, '<u4', fc, 128).astype(np.int64)
    samples = np.frombuffer(d, '<i2', sc * 3, 128 + 4 * fc).reshape(-1, 3)
    return h, table, samples


def make_block(frame, clock_us, samples, *, ring_index=0, flags=0, lost=0, take=1, readout_us=24833, mark_delay_us=17416,
               sensor_mode=3, resolution=4, dc_crop=0, bit_depth=12, axes=(2, -1, 3), lsb=131, version=1, magic=b'FPG2',
               header_bytes=48, n=None, trailer=None):
    """The block of one frame, byte for byte as in section 3 of the spec. trailer: (window w, h,
    recorded w, h) written at block offset 3056 as R124 does (section 7)."""
    samples = np.asarray(samples, np.int16).reshape(-1, 3)
    h = bytearray(48)
    struct.pack_into('<4sHHIIHHHHIIiH4B3bBH', h, 0, magic, version, header_bytes, frame & 0xFFFFFFFF, clock_us & 0xFFFFFFFF,
                     len(samples) if n is None else n, ring_index, flags, min(lost, 65535), take & 0xFFFFFFFF, readout_us,
                     mark_delay_us, sensor_mode, resolution, dc_crop, bit_depth, 0, *axes, 0, lsb)
    out = bytes(h) + samples.astype('<i2').tobytes()
    if trailer is not None:
        assert len(out) <= 3056
        out = out + bytes(3056 - len(out)) + struct.pack('<4H', *trailer)
    assert len(out) <= REGION, 'block does not fit the region'
    return out


def ifd_entries(d, off):
    n = struct.unpack_from('<H', d, off)[0]
    return {struct.unpack_from('<H', d, off + 2 + 12 * i)[0]: struct.unpack_from('<HII', d, off + 4 + 12 * i) for i in range(n)}


def layout(d):
    """(block region offset, first image-data offset) of a frame whose first bytes are d."""
    assert d[:2] == b'II'
    ifd0 = ifd_entries(d, struct.unpack_from('<I', d, 4)[0])
    exif = ifd_entries(d, ifd0[0x8769][2])
    typ, count, off = exif[0x927C]
    assert count >= 4096, 'MakerNote shorter than 4096 bytes'
    data = ifd0.get(324) or ifd0.get(273)                 # TileOffsets / StripOffsets
    first = data[2] if data[1] == 1 else struct.unpack_from('<I', d, data[2])[0]
    return off + count - REGION, first


def write_frame(src, dst, block, headers_only=False):
    """Copies src to dst with the block (None = no block, like a frame the firmware refused)."""
    with open(src, 'rb') as f:
        head = f.read(0x20000)
    at, first = layout(head)
    assert head[at:at + REGION] == bytes(REGION), f'{src}: the block region is not zero'
    if headers_only:
        data = bytearray(head[:first])
    else:
        data = bytearray(Path(src).read_bytes())
    if block is not None:
        data[at:at + len(block)] = block
    Path(dst).write_bytes(data)


def plan_blocks(h, table, samples, frames, *, no_block=(), take=1, clock0=1000000, silent_loss=()):
    """What the firmware writes for frames 0..frames-1 (None for a frame without a block).
    `frame` counts every frame, with or without a block.
    no_block: frames whose block is not written; their samples go to the next written frame, the
    newest 500 with flag bit 1 and a lost count (at most 99) when there are more.
    silent_loss: frames that get only their own slice although the frames before had no block
    (samples gone without a lost count; the clock still shows the gap)."""
    out = []
    prev = None
    for i in range(frames):
        now = int(table[i])
        clock = clock0 + int(round(now / h['rate'] * 1e6))
        if i in no_block:
            out.append(None)
            continue
        flags, lost = 0, 0
        if prev is None:
            start = max(0, now - PREROLL)
            flags |= 1
        else:
            start = int(table[i - 1]) if i in silent_loss else prev
            if now - start > CAP:
                lost = min(now - start - CAP, RING - CAP - 1)   # the ring holds 600: at most 99 can be counted
                start = now - CAP
                flags |= 2
        out.append(make_block(i, clock, samples[start:now], ring_index=(now - 1) % RING, flags=flags, lost=lost, take=take,
                              readout_us=h['readout_us'], mark_delay_us=h['mark_delay_us'], sensor_mode=h['sensor_mode'],
                              resolution=h['resolution'], dc_crop=h['dc_crop'], bit_depth=h['bit_depth'], axes=h['axes'],
                              lsb=int(round(h['lsb']))))
        prev = now
    return out


def synthesise(sources, out_dir, fpg, *, frames=None, headers_only=False, stem=None, **faults):
    """sources: the source DNG of every output frame (a list; one file may be repeated).
    Output names: <stem>_000001.DNG ... Returns the list of written paths."""
    h, table, samples = read_fpg(fpg)
    frames = min(frames or len(sources), h['frames'], len(sources))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = plan_blocks(h, table, samples, frames, **faults)
    stem = stem or Path(sources[0]).stem.rsplit('_', 1)[0]
    paths = []
    for i in range(frames):
        dst = out_dir / f'{stem}_{i + 1:06d}.DNG'
        write_frame(sources[i], dst, blocks[i], headers_only)
        paths.append(dst)
    return paths


def truncated_fpg(fpg, out, frames):
    """The first `frames` frames of a .FPG file as a file of its own (samples up to the last mark)."""
    d = Path(fpg).read_bytes()
    fc = struct.unpack_from('<I', d, 8)[0]
    table = np.frombuffer(d, '<u4', fc, 128)[:frames]
    n = int(table[-1])
    h = bytearray(d[:128])
    struct.pack_into('<II', h, 8, frames, n)
    h[63] = 0
    Path(out).write_bytes(bytes(h) + table.astype('<u4').tobytes() + d[128 + 4 * fc:128 + 4 * fc + 6 * n])
    return Path(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('clip_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--fpg')
    ap.add_argument('--frames', type=int)
    ap.add_argument('--headers-only', action='store_true')
    a = ap.parse_args()
    clip = Path(a.clip_dir)
    sources = sorted(clip.glob('*.DNG'))
    fpg = a.fpg or next(iter(sorted(clip.glob('*.FPG'))), None)
    assert sources and fpg, 'need DNG frames and a .FPG file'
    paths = synthesise(sources, a.out_dir, fpg, frames=a.frames, headers_only=a.headers_only)
    print(f'{len(paths)} frames with FPG2 blocks -> {a.out_dir}')


if __name__ == '__main__':
    main()
