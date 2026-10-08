"""Tests of the in-frame gyro path (src/gyro_inframe.cpp): "FPG2" blocks inside every DNG frame.

No camera clip with these blocks exists yet, so the frames are synthesised by
tests/make_inframe_clip.py from a .FPG track, following FPGYRO_INFRAME.md (as built in R112)
byte for byte. Where the camera project is present, the firmware's own frame hook is run in
emulation (its verify_gyro.py) and the blocks it writes are read by the plug-in's loader.

1. Without any drive: header-only frames (tests/data/frame_header.bin, the first 0x13C00 bytes
   of a real 3264x2170 frame) with blocks made from tests/data/A001_092.FPG. Block parser, track
   equal to the .FPG track, missing / damaged / capped / foreign blocks, mixed clips, cache file,
   background load and status text.
2. With the real clip (default F:/CINEMA/A001_095, read-only): full copies of its first frames
   with blocks from its own .FPG; the stabilised picture through the in-frame path equals the
   picture through the .FPG path; rendering does not wait for the scan; scan time.

usage: python tests/test_inframe.py [CLIP_DIR]
Scratch frames go to tests/scratch/ (removed at the end; --keep leaves them)."""
import json, os, shutil, struct, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import make_inframe_clip as M  # noqa: E402
from ofx_host import OFXHost, OK  # noqa: E402

# The values these tests were written for (the defaults until 1.5.1; since 1.6.0: off, 0.1 s, Dynamic).
STAB_TEST = dict(stabEnable=1, stabSmoothness=0.5, stabZoomMode=0, developRaw=1)   # and the plug-in's own development (off by default since 1.9.2)

PLUGIN = ROOT / 'dist/SigmaFpRaw.ofx.bundle/Contents/Win64/SigmaFpRaw.ofx'
CLI = ROOT / 'build/sfp_cli.exe'
SAMPLE = HERE / 'data/A001_092.FPG'
HEADER = HERE / 'data/frame_header.bin'
HEADER_R122 = HERE / 'data/frame_header_r122.bin'   # a real R122 MQ 50 frame header (A001_007), block included
SCRATCH = HERE / 'scratch'
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]
KEEP = '--keep' in sys.argv
CLIP = Path(ARGS[0] if ARGS else 'F:/CINEMA/A001_095')
CAMERA = Path(os.environ.get('SFP_CAMERA_DIR', r'C:\Users\Admin\Documents\Claude\Sigma FP Mods\analysis\recovery112_gyroframe'))
N = 292


def cli(*args):
    r = subprocess.run([str(CLI), *map(str, args)], capture_output=True, text=True)
    return json.loads(r.stdout)


def gyro(path, *opts, points=()):
    return cli('gyro', path, *opts, *[f'pt={x},{y}' for x, y in points])


def qmul(a, b):
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def relative(q, ref=0):
    """Rotation of every frame against frame `ref` (the tracks start at different samples)."""
    r = qmul(np.repeat(q[ref:ref + 1] * [1, -1, -1, -1], len(q), 0), q)
    return r * np.sign(r[:, :1])


def angle_deg(a, b):
    return np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(1)), 0, 1)))


def clip_of(name, **faults):
    """292 header-only frames with blocks from the sample .FPG; returns the first frame's path."""
    out = SCRATCH / name
    shutil.rmtree(out, ignore_errors=True)
    return M.synthesise([HEADER] * N, out, SAMPLE, headers_only=True, stem=f'{name}_20261001', **faults)


def patch_block(path, **fields):
    """Rewrites one frame's block header fields in place (damage)."""
    d = bytearray(Path(path).read_bytes())
    at, _ = M.layout(d)
    for name, value in fields.items():
        off, fmt = {'magic': (0, '4s'), 'version': (4, '<H'), 'header_bytes': (6, '<H'), 'n': (16, '<H'), 'take': (24, '<I')}[name]
        struct.pack_into(fmt, d, at + off, value)
    Path(path).write_bytes(d)


def main():
    report = []

    def ok(name, **info):
        report.append(dict(name=name, passed=True, **info))
        print('PASS', name, info if info else '')

    def skip(name, why):
        report.append(dict(name=name, skipped=why))
        print('SKIP', name, '-', why)

    tmp = Path(tempfile.mkdtemp(prefix='sfp_inframe_'))
    os.environ['SFP_GYRO_CACHE_DIR'] = str(tmp / 'cache')
    os.environ.pop('SFP_GYRO_SCAN_DELAY_US', None)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    h, table, samples = M.read_fpg(SAMPLE)

    # -- 1. block parser: every field, through the plug-in's reader --
    f = tmp / 'one.DNG'
    smp = (np.arange(3 * 77).reshape(-1, 3) * 37 % 60000 - 30000).astype(np.int16)
    M.write_frame(HEADER, f, M.make_block(123456, 0xFFFFFF00, smp, ring_index=599, flags=3, lost=4321, take=0xDEADBEEF, readout_us=24833,
                                         mark_delay_us=-17416, sensor_mode=98, resolution=7, dc_crop=1, bit_depth=14, axes=(-3, 1, -2), lsb=262),
                  headers_only=True)
    b = cli('fpg2', f)
    want = dict(present=1, frame=123456, clock_us=0xFFFFFF00, n=77, ring_index=599, flags=3, lost=4321, take=0xDEADBEEF, readout_us=24833,
                mark_delay_us=-17416, sensor_mode=98, resolution=7, dc_crop=1, bit_depth=14, axes=[-3, 1, -2], lsb_per_dps=262)
    assert {k: b[k] for k in want} == want and b['samples'] == smp.ravel().tolist(), b
    cap = (np.arange(1500) % 2000 - 1000).astype(np.int16).reshape(-1, 3)
    for n, present in ((0, 1), (500, 1)):
        M.write_frame(HEADER, f, M.make_block(1, 2, cap[:n]), headers_only=True)
        b = cli('fpg2', f)
        assert b['present'] == present and b['n'] == n and b['samples'] == cap[:n].ravel().tolist()
    bad = {'more than the 500 cap': dict(n=501), 'n beyond the region': dict(n=60000), 'version 2': dict(version=2),
           'magic': dict(magic=b'FPGX'), 'header size 40': dict(header_bytes=40), 'header size beyond the region': dict(header_bytes=4000)}
    for name, fields in bad.items():
        M.write_frame(HEADER, f, M.make_block(1, 2, cap[:100]), headers_only=True)
        patch_block(f, **fields)
        b = cli('fpg2', f)
        assert b['present'] == 0, (name, b)
    M.write_frame(HEADER, f, None, headers_only=True)
    assert cli('fpg2', f) == {'present': 0, 'why': 'no FPG2 block'}
    assert cli('fpg2', tmp / 'missing.DNG')['present'] == 0
    # Another header layout: the block is found through IFD0 -> EXIF -> MakerNote, not at the fixed offset.
    exif_at, note_at, note_len = 0x40, 0x100, 5000
    t = bytearray(note_at + note_len)
    struct.pack_into('<2sHI', t, 0, b'II', 42, 8)
    struct.pack_into('<HHHIII', t, 8, 1, 0x8769, 4, 1, exif_at, 0)
    struct.pack_into('<HHHIII', t, exif_at, 1, 0x927C, 7, note_len, note_at, 0)
    blk = M.make_block(9, 99, cap[:10], take=5)
    t[note_at + note_len - M.REGION:note_at + note_len - M.REGION + len(blk)] = blk
    (tmp / 'other_layout.DNG').write_bytes(t)
    b = cli('fpg2', tmp / 'other_layout.DNG')
    assert b['present'] == 1 and b['frame'] == 9 and b['take'] == 5 and b['n'] == 10
    struct.pack_into('<I', t, exif_at + 2 + 4, 4000)                    # MakerNote shorter than 4096 bytes
    (tmp / 'short_note.DNG').write_bytes(t)
    assert cli('fpg2', tmp / 'short_note.DNG') == {'present': 0, 'why': 'MakerNote too short'}
    ok('FPG2 block parser: all fields, 0 and 500 samples, six kinds of damage, no block, MakerNote path', damaged_kinds=len(bad))

    # -- 2. the in-frame track equals the .FPG track --
    first = clip_of('T001_001')[0]
    before = sorted(p.name for p in first.parent.iterdir())
    a = gyro(first, 'path=1', 'autozoom=0', 'frame=100', points=[(1632, 1085), (300, 300), (3000, 1900)])
    ref = gyro(SAMPLE, 'path=1', 'autozoom=0', 'frame=100', f'clip={N}', points=[(1632, 1085), (300, 300), (3000, 1900)])
    assert a['source'] == 'in-frame' and a['from_cache'] == 0 and a['blocks'] == N and a['no_data'] == 0 and a['lost_samples'] == 0
    assert (a['frames'], a['raster'], a['active'], a['fps']) == (N, [3264, 2170], [12, 5, 3240, 2160], 25.0)
    assert (a['readout_us'], a['mark_delay_us'], a['axes'], a['focal_um'], a['lsb_per_dps']) == (24800, 17000, [2, -1, 3], 28000, 131.0)
    assert a['samples'] == ref['samples'] - 100 and a['preroll'] == 500          # 500 of the sidecar's 600 pre-roll samples
    assert abs(a['rate_hz'] / ref['rate_hz'] - 1) < 1e-6, (a['rate_hz'], ref['rate_hz'])
    pa, pr = np.array(a['path']), np.array(ref['path'])
    dt = np.abs((pa[:, 0] - pa[0, 0]) - (pr[:, 0] - pr[0, 0])).max()
    track = float(angle_deg(relative(pa[:, 1:5]), relative(pr[:, 1:5])).max())
    virt = float(angle_deg(relative(pa[:, 5:9]), relative(pr[:, 5:9])).max())
    pts = float(np.abs(np.array(a['points']) - np.array(ref['points'])).max())
    assert dt < 2e-6 and track < 1e-3 and virt < 1e-3 and pts < 0.01, (dt, track, virt, pts)
    assert np.all(pa[:, 10] == 1)
    assert a['status'].startswith('On | gyro: 292 frames, in-frame, 2499.46 Hz, readout 24.8 ms, 28.0 mm = 2520 px'), a['status']
    d0 = gyro(first, 'frame=100')
    assert d0['zoom'] == gyro(SAMPLE, 'frame=100', f'clip={N}')['zoom'] and 'ends at' not in d0['status']
    ok('in-frame track = .FPG track (292 frames): rate, frame times, orientations, virtual camera, warp',
       rate_hz=a['rate_hz'], fpg_rate_hz=ref['rate_hz'], frame_time_difference_s=float(dt), orientation_difference_deg=track,
       virtual_camera_difference_deg=virt, warp_difference_px=pts)

    # -- 3. cache file: used on the next load, tied to the files, never next to the clip --
    cache = sorted((tmp / 'cache').glob('*.fpgc'))
    again = gyro(first, 'path=1', 'autozoom=0', 'frame=100')
    assert len(cache) == 1 and again['from_cache'] == 1 and again['path'] == a['path'] and again['status'] == a['status']
    assert sorted(p.name for p in first.parent.iterdir()) == before                # nothing written into the clip folder
    last = first.parent / f'T001_001_20261001_{N:06d}.DNG'
    os.utime(last, (time.time() + 5, time.time() + 5))                             # a changed file: scan again
    assert gyro(first)['from_cache'] == 0 and gyro(first)['from_cache'] == 1
    os.environ['SFP_GYRO_CACHE_DIR'] = 'off'
    assert gyro(first)['from_cache'] == 0 and len(list((tmp / 'cache').glob('*.fpgc'))) == 1
    os.environ['SFP_GYRO_CACHE_DIR'] = str(tmp / 'cache')
    ok('cache file: read on the next load, redone when a frame file changes, can be switched off, clip folder untouched',
       cache_bytes=cache[0].stat().st_size, scan_ms=a['scan_ms'], cached_load_ms=again['scan_ms'])

    # -- 4. mark clock wrapping at 2^32 microseconds --
    w = gyro(clip_of('T001_002', clock0=2 ** 32 - 3000000)[0], 'path=1', 'autozoom=0')
    assert abs(w['rate_hz'] - a['rate_hz']) < 1e-6 and w['path'] == a['path']
    ok('mark clock wrap inside the clip: same track', rate_hz=w['rate_hz'])

    # -- 5. one frame without a block: its samples are in the next slice, nothing is lost --
    m = gyro(clip_of('T001_003', no_block={50})[0], 'path=1', 'autozoom=0')
    pm = np.array(m['path'])
    assert (m['blocks'], m['no_data'], m['lost_samples'], m['samples']) == (N - 1, 0, 0, a['samples'])
    t_err = float(abs((pm[50, 0] - pm[0, 0]) - (pr[50, 0] - pr[0, 0])))
    others = np.arange(N) != 50
    assert t_err < 0.001 and np.array_equal(pm[others][:, :5], pa[others][:, :5]) and pm[50, 10] == 1
    assert angle_deg(pm[:, 5:9], pa[:, 5:9]).max() < 0.01
    ok('a frame without a block: track complete, its mark time interpolated', mark_time_error_ms=t_err * 1e3)

    # -- 6. the cap: seven frames without a block, the next one carries the newest 500 and a lost count --
    paths = clip_of('T001_004', no_block=set(range(100, 107)))
    blk = cli('fpg2', paths[107])
    lost = int(table[107] - table[99]) - 500                 # about 300 dropped; the block can only say 99
    assert blk['n'] == 500 and blk['flags'] == 2 and blk['lost'] == 99 and blk['frame'] == 107 and cli('fpg2', paths[103])['present'] == 0
    c = gyro(paths[0], 'path=1', 'frame=103', points=[(300, 300)])
    pc = np.array(c['path'])
    assert (c['blocks'], c['no_data']) == (N - 7, 7) and abs(c['lost_samples'] - lost) <= 2 and abs(c['samples'] - a['samples']) <= 2
    assert np.array_equal(pc[:, 10] == 0, (np.arange(N) >= 100) & (np.arange(N) <= 106))
    assert c['on'] == 0 and c['points'][0][2:] == [300, 300]                      # no gyro data: frame as recorded
    assert f'no gyro data on 7 of {N} frames' in c['status'] and f"{c['lost_samples']} gyro samples missing (bridged)" in c['status']
    assert np.abs((pc[:, 0] - pc[0, 0]) - (pa[:, 0] - pa[0, 0]))[pc[:, 10] == 1].max() < 0.001  # the clock keeps running over the gap
    after = angle_deg(relative(pc[110:, 1:5]), relative(pa[110:, 1:5])).max()                 # same motion after the gap
    assert after < 1e-3, after
    z = pc[:, 9]
    assert z[99] == 1 and z[107] == 1 and np.all(z[100:107] == 1) and z[60] == z[0] > 1.2 and z[140] == z[0]
    assert np.all(np.diff(z[74:100]) <= 0) and np.all(np.diff(z[107:133]) >= 0)               # fades out before, in after
    near = gyro(paths[0], 'frame=99', points=[(300, 300)])
    assert near['on'] == 1 and max(abs(near['points'][0][2] - 300), abs(near['points'][0][3] - 300)) < 0.01
    # A short overflow: the lost count is exact (one frame without a block and a late one: 5 slices + 50).
    ok('capped slice: newest 500 samples, flag bit 1, lost 99 at most; the gap length comes from the mark clock; frames without data not stabilised, fade around them',
       dropped=lost, lost_field=blk['lost'], gap_in_track=c['lost_samples'], zoom_around_gap=[float(v) for v in z[[74, 87, 95, 99, 103, 107, 111, 119, 132]]])

    # -- 6b. a frame number without a file (A001_096 has no frame 1260): not a frame "without gyro data" --
    paths = clip_of('T001_013')
    paths[200].unlink()
    paths[N - 2].unlink()
    r = gyro(paths[0], 'path=1')
    pg = np.array(r['path'])
    assert (r['frames'], r['blocks'], r['no_data']) == (N, N - 2, 0) and 'no gyro data' not in r['status'] and np.all(pg[:, 10] == 1)
    assert abs(r['lost_samples'] - int(table[200] - table[199]) - int(table[N - 2] - table[N - 3])) <= 2 and 'gyro samples missing (bridged)' in r['status']
    assert np.all(pg[150:260, 9] == pg[0, 9]) and pg[N - 1, 9] == pg[0, 9]                      # no fade around the missing numbers
    assert np.abs((pg[:, 0] - pg[0, 0]) - (pa[:, 0] - pa[0, 0])).max() < 0.001
    ok('frame numbers without a file: their samples are bridged, the frames around them stay stabilised', bridged_samples=r['lost_samples'])

    # -- 6c. late frame marks (the recorder task held up): the frame times stay on the regular cadence --
    late_table = table.copy()
    for k, v in {60: 55, 61: 12, 150: 92, 151: 44, 152: 3}.items():
        late_table[k] += v
    out = SCRATCH / 'T001_014'
    out.mkdir()
    blocks = M.plan_blocks(h, late_table, samples, N)
    paths = [out / f'T001_014_20261001_{i + 1:06d}.DNG' for i in range(N)]
    for pth, blk in zip(paths, blocks):
        M.write_frame(HEADER, pth, blk, headers_only=True)
    r = gyro(paths[0], 'path=1', 'autozoom=0')
    pl = np.array(r['path'])
    assert r['late_marks'] == 3 and r['dropped_frames'] == 0 and '| 3 late frame marks corrected' in r['status'], (r['late_marks'], r['status'])
    assert np.abs((pl[:, 0] - pl[0, 0]) - (pa[:, 0] - pa[0, 0])).max() < 0.0005                 # same frame times as without the late marks
    raw = np.array(gyro(paths[0], 'path=1', 'autozoom=0', 'marks=raw')['path'])
    assert abs((raw[150, 0] - raw[100, 0]) - (pa[150, 0] - pa[100, 0]) - 92 / 2499.46) < 0.0015  # what each frame's own mark would give
    ok('late frame marks in the blocks: frame times stay on the cadence', late_marks=r['late_marks'], worst_late_ms=92 / 2.49946,
       largest_time_error_ms=float(1e3 * np.abs((pl[:, 0] - pl[0, 0]) - (pa[:, 0] - pa[0, 0])).max()))

    # -- 6d. exposure time from the DNG: sync = readout/2 + exposure/2, also when it changes inside the clip --
    assert a['delay_from_exposure'] == 1 and abs(a['delay_ms'] - 17.4) < 1e-6 and '| sync 17.4 ms = readout/2 + exposure/2 (1/100 s) |' in a['status'], a['status']
    paths = clip_of('T001_015')
    for pth in paths[N // 2:]:                                 # second half of the clip at 1/50 s
        d = bytearray(pth.read_bytes())
        exif = M.ifd_entries(d, M.ifd_entries(d, struct.unpack_from('<I', d, 4)[0])[0x8769][2])
        struct.pack_into('<II', d, exif[0x829A][2], 1, 50)
        pth.write_bytes(d)
    r0, r1 = gyro(paths[0], 'frame=0'), gyro(paths[0], f'frame={N - 1}')
    assert abs(r0['delay_ms'] - 17.4) < 1e-3 and abs(r1['delay_ms'] - 22.4) < 1e-3 and 'the exposure changes in the clip' in r0['status'], (r0['delay_ms'], r1['delay_ms'])
    ok('sync from the EXIF exposure time; a change of exposure inside the clip is followed', delay_ms_first_last=[r0['delay_ms'], r1['delay_ms']])

    # -- 7. damaged blocks --
    paths = clip_of('T001_005')
    patch_block(paths[60], n=600)
    patch_block(paths[61], version=9)
    patch_block(paths[62], header_bytes=20)
    dm = gyro(paths[0], 'path=1')
    pd = np.array(dm['path'])
    assert (dm['blocks'], dm['no_data']) == (N - 3, 3) and np.array_equal(np.where(pd[:, 10] == 0)[0], [60, 61, 62])
    assert abs(dm['lost_samples'] - int(table[62] - table[59])) <= 2             # their samples are gone: the clock shows it
    assert np.abs((pd[:, 0] - pd[0, 0]) - (pa[:, 0] - pa[0, 0])).max() < 0.001
    ok('damaged blocks are skipped; the mark clock gives the length of the gap', gap_samples=dm['lost_samples'])

    # -- 8. samples gone without a lost count (blocks missing for longer than the ring holds) --
    paths = clip_of('T001_006', no_block=set(range(150, 171)), silent_loss={171})
    sl = gyro(paths[0], 'path=1')
    ps = np.array(sl['path'])
    assert sl['no_data'] == 21 and abs(sl['lost_samples'] - int(table[170] - table[149])) <= 2
    assert np.abs((ps[:, 0] - ps[0, 0]) - (pa[:, 0] - pa[0, 0]))[np.r_[0:150, 171:N]].max() < 0.001
    assert abs(sl['rate_hz'] / a['rate_hz'] - 1) < 1e-4
    ok('samples missing without a lost count: gap from the mark clock, timing after it intact', gap_samples=sl['lost_samples'], rate_hz=sl['rate_hz'])

    # -- 9. mixed clips --
    mixed = {}
    # frames of another take inside the sequence
    paths = clip_of('T001_007')
    other = M.plan_blocks(h, table, samples, N, take=2)
    for i in range(200, 220):
        M.write_frame(HEADER, paths[i], other[i], headers_only=True)
    r = gyro(paths[0], 'path=1')
    assert r['no_data'] == 20 and np.array_equal(np.where(np.array(r['path'])[:, 10] == 0)[0], np.arange(200, 220))
    mixed['other_take_frames_without_data'] = r['no_data']
    # the first 30 frames carry no block (an older stretch): the take starts with frame 31
    paths = clip_of('T001_008', no_block=set(range(30)))
    r = gyro(paths[0], 'path=1')
    p8 = np.array(r['path'])
    assert cli('fpg2', paths[30])['flags'] == 1 and r['no_data'] == 30 and r['blocks'] == N - 30 and np.all(p8[:30, 10] == 0)
    assert p8[30, 9] == 1 and np.all(np.diff(p8[30:56, 9]) >= 0) and p8[60, 9] > 1.2
    assert f'no gyro data on 30 of {N} frames' in r['status']
    mixed['leading_frames_without_data'] = r['no_data']
    # no block anywhere, a .FPG next to the frames: the sidecar is used, as before
    paths = clip_of('T001_009', no_block=set(range(N)))
    shutil.copy(SAMPLE, paths[0].parent / 'T001_009_20261001.FPG')
    r = gyro(paths[0])
    assert r['source'] == 'fpg' and r['status'].startswith('On | T001_009_20261001.FPG: 292 frames, 2499.46 Hz'), r
    # blocks in the frames and a .FPG next to them: the frames win
    paths = clip_of('T001_010')
    shutil.copy(SAMPLE, paths[0].parent / 'T001_010_20261001.FPG')
    assert gyro(paths[0])['source'] == 'in-frame'
    # neither
    paths = clip_of('T001_011', no_block=set(range(N)))
    assert gyro(paths[0])['error'] == 'no gyro data in the DNGs, no .FPG gyro file next to the DNGs' 
    # a single frame with a block cannot give a sample rate
    paths = clip_of('T001_012', no_block=set(range(N)) - {42})
    assert 'fewer than two frames carry gyro data' in gyro(paths[0])['error']
    ok('mixed clips: other take, leading frames without blocks, sidecar fallback, frames before sidecar, nothing, one block', **mixed)

    # -- 9a. image scale from the sensor window (R122, R123, R124) --
    window_tests(tmp, ok, h, table, samples)

    # -- 9a2. real frames on the camera SSD (read-only): binning per clip, a damaged frame --
    e_drive_checks(ok, skip)

    # -- 9b. blocks written by the firmware itself (R112 frame hook, emulated) --
    firmware_check(tmp, ok, skip)

    # -- 10. background load and status text in the plug-in --
    background_status(ok)
    os.environ['SFP_GYRO_CACHE_DIR'] = str(tmp / 'cache')

    # -- 11. real frames --
    frames = sorted(CLIP.glob('*.DNG')) if CLIP.is_dir() else []
    fpgs = sorted(CLIP.glob('*.FPG')) if CLIP.is_dir() else []
    if len(frames) < 120 or not fpgs:
        skip('real frames: same picture through both paths, render during the scan, scan time', f'{CLIP} with its .FPG is not connected')
    else:
        real_frames(frames, fpgs[0], tmp, ok)

    # -- 12. scan time on header-only frames --
    big = SCRATCH / 'T001_scan'
    hb, tb, sb = M.read_fpg(fpgs[0]) if fpgs else (h, table, samples)
    count = min(1000, hb['frames'])
    paths = M.synthesise([HEADER] * count, big, fpgs[0] if fpgs else SAMPLE, headers_only=True, stem='T001_scan_20261001')
    s1 = cli('fpg2scan', paths[0])
    os.environ['SFP_GYRO_CACHE_DIR'] = 'off'
    full = gyro(paths[0])
    os.environ['SFP_GYRO_CACHE_DIR'] = str(tmp / 'cache')
    assert s1['blocks'] == count and full['blocks'] == count and full['scan_ms'] < 5000
    ok('scan time, header-only frames on the system SSD (first read of new files, then file cache warm)', frames=count,
       first_read_of_new_files_ms=s1['ms'], whole_load_ms=full['scan_ms'],
       ms_per_1000_frames=full['scan_ms'] * 1000 / count, extrapolated_90000_frames_s=full['scan_ms'] * 90 / count)

    if not KEEP:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    (ROOT / 'tests/inframe_report.json').write_text(json.dumps({'plugin': str(PLUGIN), 'clip': str(CLIP), 'tests': report}, indent=2) + '\n')
    done = sum(1 for r in report if r.get('passed'))
    print(f'ALL PASS ({done})' + (f', {len(report) - done} skipped' if done != len(report) else ''))


R124_MODES = [   # mode, DC, rates, window, recorded, crop code, readout us (FPGYRO_INFRAME.md section 7)
    ('3:2', 0, '<=29.97', (6000, 4000), (3840, 2560), 0, 24700), ('3:2', 1, '<=29.97', (4024, 2682), (3840, 2560), 1, 16600),
    ('3:2', 1, '48/50', (4024, 2682), (3000, 2000), 1, 16600), ('16:9', 0, '<=29.97', (6000, 3375), (3840, 2160), 0, 20900),
    ('16:9', 1, '<=29.97', (4216, 2372), (3840, 2160), 1, 14700), ('16:9', 1, '48/50', (4216, 2372), (3200, 1800), 1, 14700),
    ('2:1', 0, '<=29.97', (6000, 3000), (4320, 2160), 0, 18500), ('2:1', 0, '48/50', (6000, 3000), (3600, 1800), 0, 18500),
    ('2:1', 1, '<=29.97', (4320, 2160), (4320, 2160), 1, 13400), ('2:1', 1, '48/50', (4320, 2160), (3600, 1800), 1, 13400),
    ('M43', 0, '<=50', (3000, 2000), (3000, 2000), 2, 12400), ('S16', 0, '<=50', (2288, 1526), (2160, 1440), 2, 9400)]


def small_clip(name, header, frames=12, **block):
    """A few header-only frames with blocks made from the sample track; block = make_block fields."""
    h, table, samples = M.read_fpg(SAMPLE)
    out = SCRATCH / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    paths = []
    for i in range(frames):
        start = int(table[i - 1]) if i else int(table[0]) - 500
        blk = M.make_block(i, 1000000 + int(round(table[i] / h['rate'] * 1e6)), samples[start:int(table[i])], flags=0 if i else 1, **block)
        d = bytearray(Path(header).read_bytes())
        at, _ = M.layout(d)
        d[at:at + M.REGION] = bytes(M.REGION)                 # a fixture may carry a block of its own
        d[at:at + len(blk)] = blk
        paths.append(out / f'{name}_20261003_{i + 1:06d}.DNG')
        paths[-1].write_bytes(d)
    return paths


def window_tests(tmp, ok, h, table, samples):
    os.environ['SFP_GYRO_CACHE_DIR'] = 'off'
    # R124: the trailer carries the window and the recorded size.
    rows = {}
    for k, (mode, dc, rate, win, rec, crop, ro) in enumerate(R124_MODES):
        paths = small_clip(f'W124_{k:02d}', HEADER, readout_us=ro, mark_delay_us=ro // 2 + 5000, sensor_mode=121 if rate == '48/50' else 3,
                           resolution={'3:2': 6, '16:9': 3, '2:1': 4, 'M43': 2, 'S16': 8}[mode], dc_crop=crop, trailer=(*win, *rec))
        b = cli('fpg2', paths[1])
        assert (b['window'], b['recorded'], b['dc_crop'], b['readout_us']) == (list(win), list(rec), crop, ro), b
        j = gyro(paths[0])
        want = 28.0 / (0.006 * win[0] / rec[0])
        assert j['window'] == list(win) and j['recorded'] == list(rec) and j['window_source'] == 1 and abs(j['focal_px'] - want) < 1e-3, (mode, j['focal_px'], want)
        assert (f'(sensor window {win[0]}x{win[1]})' in j['status']) == (win[0] != 6000), j['status']
        rows[f'{mode} DC{dc} {rate}'] = round(j['focal_px'], 1)
    ok('R124: image scale from the window and recorded size in the block, every mode and rate group (focal px at 28 mm)', **rows)
    # R123 (crop code 2) and R122 (crop code 0, readout 18926 us) MQ 50 frames: the 4608 x 3072 window.
    r122 = cli('fpg2', HEADER_R122)
    assert (r122['present'], r122['readout_us'], r122['sensor_mode'], r122['resolution'], r122['dc_crop'], r122['window']) == (1, 18926, 121, 2, 0, [0, 0]), r122
    found = {}
    for name, crop, ro, want_src, want_w in (('W123', 2, 18926, 2, 4608), ('W122', 0, 18926, 3, 4608), ('WMQ', 0, 12423, 0, 6000)):
        paths = small_clip(name, HEADER_R122, readout_us=ro, mark_delay_us=ro // 2 + 5000, sensor_mode=121 if ro == 18926 else 98,
                           resolution=2, dc_crop=crop)
        j = gyro(paths[0])
        assert j['raster'] == [3024, 2010] and j['recorded'] == [3000, 2000], j
        assert j['window'][0] == want_w and j['window_source'] == want_src and abs(j['focal_px'] - 28 / (0.006 * want_w / 3000)) < 1e-3, (name, j['window'], j['focal_px'])
        found[name] = (j['window'], round(j['focal_px'], 1), j['status'])
    assert '(sensor window 4608x3072, MQ 50, recognised by its readout)' in found['W122'][2]
    assert '(sensor window 4608x3072, MQ 50),' in found['W123'][2] and 'sensor window' not in found['WMQ'][2]
    # An older DC Crop block (crop code 1, no trailer): the 1.49 crop.
    paths = small_clip('WDC', HEADER, readout_us=20000, mark_delay_us=15000, dc_crop=1)
    j = gyro(paths[0])
    assert j['window'] == [4024, 2682] and abs(j['focal_px'] - 28 / (0.006 * 4024 / 3240)) < 1e-3 and '(sensor window 4024x2682, DC Crop)' in j['status']
    ok('R123 (crop code 2) and R122 (readout 18926 us + MQ size) give the 4608 x 3072 window; plain MQ and older DC Crop as before',
       r123=found['W123'][:2], r122=found['W122'][:2], plain_mq=found['WMQ'][:2], older_dc_crop=[j['window'], round(j['focal_px'], 1)])
    # Row Phase and Edge Anti-aliasing: only for the 2x2 binned readout (M98), not for 1:1 windows of the same 3024x2010 size.
    binning = {}
    for name, header, block, want in (
            ('B122', HEADER_R122, dict(readout_us=18926, sensor_mode=121, resolution=2, dc_crop=0), 0),
            ('B123', HEADER_R122, dict(readout_us=18926, sensor_mode=121, resolution=2, dc_crop=2), 0),
            ('B124', HEADER_R122, dict(readout_us=12400, sensor_mode=3, resolution=2, dc_crop=2, trailer=(3000, 2000, 3000, 2000)), 0),
            ('B124DC', HEADER_R122, dict(readout_us=16600, sensor_mode=121, resolution=6, dc_crop=1, trailer=(4024, 2682, 3000, 2000)), 0),
            ('BM98', HEADER_R122, dict(readout_us=12423, sensor_mode=98, resolution=2, dc_crop=0), 1),
            ('BHQ', HEADER, dict(readout_us=24833, sensor_mode=3, resolution=4, dc_crop=0), 0)):
        j = gyro(small_clip(name, header, **block)[0])
        assert j['binned'] == want, (name, j['binned'], j['binned_why'])
        binning[name] = f"{j['binned']}: {j['binned_why']}"
    ok('Row Phase + Edge Anti-aliasing decided from the gyro block: on for the 2x2 binned M98 readout only', **binning)

    # The real R122 clip, if the SSD is connected.
    clip = Path(os.environ.get('SFP_R122_CLIP', 'E:/CINEMA/A001_007'))
    frames = sorted(clip.glob('*.DNG')) if clip.is_dir() else []
    if len(frames) < 100:
        print('SKIP real R122 clip -', clip, 'is not connected')
    else:
        j = gyro(frames[0])
        assert (j['window'], j['window_source'], j['readout_us'], j['sensor_mode']) == ([4608, 3072], 3, 18926, 121), j
        assert abs(j['focal_px'] - 28 / (0.006 * 4608 / 3000)) < 1e-3 and '(sensor window 4608x3072, MQ 50, recognised by its readout)' in j['status']
        ok(f'{clip.name} (R122, MQ 50): window recognised from its blocks', frames=j['frames'], focal_px=j['focal_px'], status=j['status'][:150])
    os.environ['SFP_GYRO_CACHE_DIR'] = str(tmp / 'cache')


def e_drive_checks(ok, skip):
    root = Path(os.environ.get('SFP_E_CINEMA', 'E:/CINEMA'))
    clips = {n: sorted((root / n).glob('*.DNG')) for n in ('A001_004', 'A001_005', 'A001_007')}
    if any(len(v) < 50 for v in clips.values()):
        return skip('camera SSD clips: binning and a damaged frame', f'{root} does not hold A001_004 / A001_005 / A001_007')
    os.environ['SFP_GYRO_CACHE_DIR'] = 'off'

    def host_for(frames):
        host = OFXHost(PLUGIN, canvas=(1504, 1000))
        host.set(**STAB_TEST)
        source(host, frames[0])
        host.set(decodeQuality=1, stabEnable=0)
        return host
    res = {}
    for name, on in (('A001_004', True), ('A001_007', False)):
        host = host_for(clips[name])
        info = host.get('info')
        st, a = host.render(source_frame=20)
        host.set(rowPhase=0.0, deZigzag=0.0)
        st2, b = host.render(source_frame=20)
        host.close()
        assert st == OK and st2 == OK and (not np.array_equal(a.pixels, b.pixels)) == on, (name, info)
        assert info.endswith('Row Phase + Edge Anti-aliasing on: 2x2 binned readout (M98)' if on else
                             'Row Phase + Edge Anti-aliasing off: 1:1 sensor window (MQ 50, R122 readout)'), info
        res[name] = info.split(' | ')[-1]
    ok('A001_004 (old MQ, M98): corrections on; A001_007 (R122 MQ 50 window): off, the picture equals Row Phase 0 / Anti-aliasing 0', **res)
    # A001_005 frame 2: tile 4 of its LJ92 data is damaged in camera.
    host = host_for(clips['A001_005'])
    st, bad = host.render(source_frame=1)
    msg = host.messages[-1]
    st2, prev = host.render(source_frame=0)
    host.close()
    assert st == OK and st2 == OK and np.array_equal(bad.pixels, prev.pixels), host.messages[-3:]
    assert msg['type'] == 'OfxMessageWarning' and 'frame 2 cannot be decoded (Tile 4: LJ92:' in msg['text'] and 'showing frame 1 instead' in msg['text'], msg
    r = subprocess.run([str(CLI), 'develop', str(clips['A001_005'][1]), str(Path(tempfile.gettempdir()) / 'sfp_bad.tif'), 'width=64', 'height=43'],
                       capture_output=True, text=True)
    assert 'damaged data' in r.stdout, r.stdout
    ok('A001_005 frame 2 (damaged LJ92 tile in camera): the plug-in shows frame 1 with a warning instead of failing', warning=msg['text'])


def firmware_frames():
    """Runs the camera project's verify_gyro.py with every frame header its emulated firmware
    hook produced captured: a list of runs (one per emulated power-on), each a list of headers."""
    src = (CAMERA / 'verify_gyro.py').read_text()
    a = "        return dict(hdr=after[RESERVE-H:RESERVE],"
    b = "        u.mem_write(0x44E00000,bytes([0xEE])*0x400000)"
    assert src.count(a) == 1 and src.count(b) == 1
    src = src.replace(a, "        CAPTURED[-1].append(after[RESERVE-H:RESERVE])\n" + a).replace(b, "        CAPTURED.append([])\n" + b)
    ns = {'__name__': 'verify_gyro_capture', '__file__': str(CAMERA / 'verify_gyro.py'), 'CAPTURED': []}
    exec(compile(src, str(CAMERA / 'verify_gyro.py'), 'exec'), ns)
    cases = ns['run'](CAMERA / 'joined/MAIN.r45.joined.ramresearch')
    return ns['CAPTURED'], len(cases)


def firmware_check(tmp, ok, skip):
    name = 'blocks written by the R112 firmware (emulated) are read by the plug-in'
    if not (CAMERA / 'verify_gyro.py').exists() or not (CAMERA / 'joined/MAIN.r45.joined.ramresearch').exists():
        return skip(name, f'{CAMERA} is not present')
    try:
        import unicorn, capstone  # noqa: F401
    except ImportError:
        return skip(name, 'unicorn / capstone are not installed')
    runs, cases = firmware_frames()
    steps = [104, 104, 105, 21, 20, 21, 208, 100, 1, 0, 0, 3, 417, 499, 500, 100, 99, 101] + [100 + (i * 37) % 9 for i in range(42)]
    seq = next(r for r in runs if len(r) == 61)                       # "exactly once": 61 frames, slices of 0..500 samples
    cap = next(r for r in runs if len(r) == 4 and cli_block(tmp, r[1])['lost'] == 50)

    def smp(k):                                                       # the emulated gyro: sample k is (k, k + 0x4000, -k) as i16
        return [((v & 0xFFFF) ^ 0x8000) - 0x8000 for v in (k, k + 0x4000, -k)]

    def write(run, stem):
        d = SCRATCH / stem
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        out = []
        for i, hdr in enumerate(run):
            out.append(d / f'{stem}_20261001_{i + 1:06d}.DNG')
            out[-1].write_bytes(hdr)
        return out
    paths = write(seq, 'FW01_001')
    blocks = [cli('fpg2', p) for p in paths]
    assert all(b['present'] for b in blocks) and [b['frame'] for b in blocks] == list(range(61))
    assert [b['n'] for b in blocks] == [500] + steps and blocks[0]['flags'] == 1 and all(b['flags'] == 0 and b['lost'] == 0 for b in blocks[1:])
    b0 = blocks[0]
    assert (b0['take'], b0['axes'], b0['lsb_per_dps'], b0['resolution'], b0['dc_crop'], b0['bit_depth']) == (1, [2, -1, 3], 131, 4, 0, 12), b0
    assert b0['mark_delay_us'] == b0['readout_us'] // 2 + 5000 and b0['readout_us'] > 0
    stream = [v for b in blocks for v in b['samples']]
    first = 1000 - 499
    assert stream == [v for k in range(first, first + 500 + sum(steps)) for v in smp(k)]
    os.environ['SFP_GYRO_CACHE_DIR'] = 'off'
    j = gyro(paths[0], 'path=1')
    t = np.array(j['path'])[:, 11]                             # the marks as recorded (these frames have no regular cadence)
    assert j['source'] == 'in-frame' and (j['frames'], j['blocks'], j['no_data'], j['lost_samples'], j['samples']) == (61, 61, 0, 0, 500 + sum(steps))
    assert abs(j['rate_hz'] - 2500.0) < 1e-6 and (j['readout_us'], j['mark_delay_us'], j['axes']) == (b0['readout_us'], b0['mark_delay_us'], [2, -1, 3])
    assert np.abs((t - t[0]) - np.cumsum([0] + steps) / 2500.0).max() < 1e-8
    # The cap: 550 waiting -> 500 + lost 50; 599 waiting -> 500 + lost 99.
    paths = write(cap, 'FW01_002')
    cb = [cli('fpg2', p) for p in paths]
    assert [(b['n'], b['flags'], b['lost'], b['frame']) for b in cb] == [(500, 1, 0, 0), (500, 2, 50, 1), (500, 2, 99, 2), (100, 0, 0, 3)], cb
    c = gyro(paths[0], 'path=1')
    tc = np.array(c['path'])[:, 11]
    assert (c['blocks'], c['no_data'], c['lost_samples'], c['samples']) == (4, 0, 149, 1749) and abs(c['rate_hz'] - 2500.0) < 1e-6
    assert np.abs((tc - tc[0]) - np.array([0, 550, 1149, 1249]) / 2500.0).max() < 1e-8
    # The synthesiser writes the same bytes as the firmware for the same input.
    b1 = blocks[1]
    mine = M.make_block(b1['frame'], b1['clock_us'], np.array(b1['samples']).reshape(-1, 3), ring_index=b1['ring_index'], flags=b1['flags'],
                        lost=b1['lost'], take=b1['take'], readout_us=b1['readout_us'], mark_delay_us=b1['mark_delay_us'],
                        sensor_mode=b1['sensor_mode'], resolution=b1['resolution'], dc_crop=b1['dc_crop'], bit_depth=b1['bit_depth'],
                        axes=b1['axes'], lsb=b1['lsb_per_dps'])
    at, _ = M.layout(seq[1])
    assert at == 0x10D56 and seq[1][at:at + M.REGION] == mine + bytes(M.REGION - len(mine))
    os.environ['SFP_GYRO_CACHE_DIR'] = str(tmp / 'cache')
    ok(name, firmware_verifier_cases=cases, frames=61, samples=j['samples'], rate_hz=j['rate_hz'], readout_us=b0['readout_us'],
       capped_frames_lost=[b['lost'] for b in cb], synthesiser_bytes_equal_firmware=True)


def cli_block(tmp, header):
    f = tmp / 'fw_probe.DNG'
    f.write_bytes(header)
    return cli('fpg2', f)


def source(host, first):
    host.obj(host.obj(host.instance)['props'])['values']['OfxImageEffectPropSrcFilePath'] = [str(first)]
    host.changed('Source')


def wait_ready(host, seconds=30):
    seen = []
    end = time.time() + seconds
    while time.time() < end:
        host.changed('stabEnable')
        s = host.get('stabStatus')
        if not seen or seen[-1] != s:
            seen.append(s)
        if not s.startswith('Loading'):
            return seen
        time.sleep(0.05)
    raise AssertionError(seen)


def background_status(ok):
    first = clip_of('T001_020')[0]
    os.environ['SFP_GYRO_CACHE_DIR'] = 'off'
    os.environ['SFP_GYRO_SCAN_DELAY_US'] = '250000'          # as if the frames were on slow media: about 2 s for the clip
    host = OFXHost(PLUGIN, canvas=(1620, 1080))
    host.set(**STAB_TEST)
    t0 = time.time()
    source(host, first)
    took = time.time() - t0
    status = host.get('stabStatus')
    assert status.startswith('Loading gyro ') and '% (292 frames) | not stabilised until it is ready' in status, status
    assert (host.get('stabFocal'), host.get('stabReadout')) == (0.0, 0.0)          # nothing to prefill yet
    assert took < 1.5, took                                  # the host is not held for the scan
    seen = wait_ready(host)
    percents = [int(s.split()[2]) for s in seen if s.startswith('Loading')]
    assert percents == sorted(percents) and percents[-1] > 0, seen
    assert seen[-1].startswith('On | gyro: 292 frames, in-frame, 2499.46 Hz, readout 24.8 ms, 28.0 mm = 2520 px, zoom 1.300'), seen[-1]
    assert (host.get('stabFocal'), host.get('stabReadout')) == (28.0, 24.8)        # prefilled once the data is there
    host.set(stabEnable=0)
    host.changed('stabEnable')
    assert host.get('stabStatus').startswith('Off (switched off) | gyro: 292 frames, in-frame')
    host.close()
    os.environ.pop('SFP_GYRO_SCAN_DELAY_US')
    # Second instance on the same clip in the same session after an unload: scanned again, at once without the delay.
    host = OFXHost(PLUGIN, canvas=(1620, 1080))
    host.set(**STAB_TEST)
    source(host, first)
    assert host.get('stabStatus').startswith('On | gyro: 292 frames, in-frame'), host.get('stabStatus')
    host.close()
    ok('plug-in: status shows "Loading gyro N %" while the scan runs in the background, then the in-frame line',
       host_blocked_s=took, percent_seen=percents, final=seen[-1])


def real_frames(frames, fpg, tmp, ok):
    count = 120
    out = SCRATCH / 'A001_095_inframe'
    paths = M.synthesise(frames[:count], out, fpg, frames=count)
    short = M.truncated_fpg(fpg, tmp / 'first120.FPG', count)
    os.environ['SFP_GYRO_CACHE_DIR'] = 'off'
    picks = (0, 30, 60, 119)

    def render(**values):
        host = OFXHost(PLUGIN, canvas=(1620, 1080))
        host.set(**STAB_TEST)
        source(host, paths[0])
        host.set(decodeQuality=1, **values)
        host.changed('gyroFile')
        seen = wait_ready(host)
        pics = []
        for i in picks:
            st, img = host.render(source_frame=i)
            assert st == OK, host.messages[-2:]
            pics.append(img.pixels.copy())
        host.close()
        return pics, seen[-1]
    by_fpg, s_fpg = render(gyroFile=str(short))
    by_frames, s_in = render()
    plain, _ = render(stabEnable=0)
    assert s_fpg.startswith('On | first120.FPG: 120 frames') and s_in.startswith('On | gyro: 120 frames, in-frame'), (s_fpg, s_in)
    diff = [float(np.abs(a - b).max()) for a, b in zip(by_fpg, by_frames)]
    moved = [float(np.abs(a - b).mean()) for a, b in zip(by_frames, plain)]
    assert max(diff) < 2e-3 and min(moved) > 1e-3, (diff, moved)
    ja, jf = gyro(paths[0], 'path=1'), gyro(short, 'path=1', f'clip={count}')
    track = float(angle_deg(relative(np.array(ja['path'])[:, 5:9]), relative(np.array(jf['path'])[:, 5:9])).max())
    assert ja['zoom'] == jf['zoom'] and track < 1e-3
    ok('A001_095, 120 real frames with synthesised blocks: stabilised picture through the frames = through the .FPG',
       frames_compared=list(picks), max_pixel_difference=diff, mean_change_against_unstabilised=moved, zoom=ja['zoom'],
       virtual_camera_difference_deg=track)

    # Rendering does not wait for the scan: frames are unstabilised until the track is ready.
    os.environ['SFP_GYRO_SCAN_DELAY_US'] = '500000'
    host = OFXHost(PLUGIN, canvas=(1620, 1080))
    host.set(**STAB_TEST)
    source(host, paths[0])
    host.set(decodeQuality=1)
    t0 = time.time()
    st, early = host.render(source_frame=60)
    took = time.time() - t0
    status = host.get('stabStatus')
    assert st == OK and status.startswith('Loading gyro') and np.array_equal(early.pixels, plain[2]) and took < 1.5, (status, took)
    seen = wait_ready(host)
    st, late = host.render(source_frame=60)
    assert st == OK and np.array_equal(late.pixels, by_frames[2])
    host.close()
    os.environ.pop('SFP_GYRO_SCAN_DELAY_US')
    ok('render during the background scan: unstabilised at once, stabilised when the track is ready',
       first_render_s=took, status_then=status, status_after=seen[-1][:60])

    # Scan time on real, full-size frames.
    s = cli('fpg2scan', paths[0])
    full = gyro(paths[0])
    assert s['blocks'] == count and full['from_cache'] == 0
    ok('scan time, 120 full-size frames on the system SSD (file cache warm; about 20 ms of it is fixed)', read_blocks_ms=s['ms'],
       whole_load_ms=full['scan_ms'])
    os.environ['SFP_GYRO_CACHE_DIR'] = str(tmp / 'cache')


if __name__ == '__main__':
    main()
