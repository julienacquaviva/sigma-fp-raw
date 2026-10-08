"""Tests of the gyro stabilisation (src/gyro.cpp, the warp in kernel_params.h / kernels.cu).

1. Unit tests on synthetic .FPG files written here from the format spec, and on the real
   sample tests/data/A001_092.FPG, through `sfp_cli gyro` (the same code the plug-in runs).
   The expected values come from an independent numpy model of the specification.
2. The built .ofx in the independent OFX host: status text, no file = unchanged image,
   synthetic gyro files against a real frame on the GPU, automatic pick-up of the sidecar.
3. If the real clips are connected: frame-to-frame image shift of frames 1..290 of A001_092,
   stabilised against unstabilised; rolling-shutter direction and amount on A001_092/093/094;
   the fade-out at the end of the gyro data on A001_093 (camera-written file, found by itself).

usage: python tests/test_gyro.py [CLIP_DIR]     (default F:/CINEMA/A001_092, read-only)
Parts 2 and 3 are skipped (and reported as skipped) without the clip."""
import json, os, shutil, struct, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ['SFP_GYRO_CACHE_DIR'] = 'off'                    # no cache files from these runs
sys.path.insert(0, str(HERE))
from ofx_host import OFXHost, OK  # noqa: E402

# The values these tests were written for (the defaults until 1.5.1; since 1.6.0: off, 0.1 s, Dynamic).
STAB_TEST = dict(stabEnable=1, stabSmoothness=0.5, stabZoomMode=0, developRaw=1)   # and the plug-in's own development (off by default since 1.9.2)

PLUGIN = ROOT / 'dist/SigmaFpRaw.ofx.bundle/Contents/Win64/SigmaFpRaw.ofx'
CLI = ROOT / 'build/sfp_cli.exe'
SAMPLE = HERE / 'data/A001_092.FPG'
CLIP = Path(sys.argv[1] if len(sys.argv) > 1 else 'F:/CINEMA/A001_092')

RASTER = (3264, 2170)
ACTIVE = (12, 5, 3240, 2160)
RATE = 2500.0
READOUT = 0.0248
MARK_DELAY = 0.017
PITCH_NM = 10000            # synthetic files: 28 mm -> 2800 px
F_PX = 2800.0
CX, CY = RASTER[0] / 2, RASTER[1] / 2
SENSOR_PITCH_MM = 0.006     # the fp's sensor: 6000 pixels over 36 mm; a frame without a window covers its whole width


# ---- synthetic files and the reference model ----

def write_fpg(path, counts, frames, *, rate_mhz=2500000, lsb=131.0, raster=RASTER, active=ACTIVE, readout_us=24800,
              focal_um=28000, mark_delay_us=17000, axes=(2, -1, 3), flags=0, preroll=600, pitch_nm=PITCH_NM,
              version=1, magic=b'FPGY', exposure_us=0, table=None):
    """counts: (samples, 3) int16 gyro X, Y, Z. Frame i's mark is after preroll + 100 * i samples
    (or `table`). exposure_us 0 = unknown: the file's mark delay is used as it is."""
    counts = np.asarray(counts, np.int16)
    table = (preroll + 100 * np.arange(frames)).astype('<u4') if table is None else np.asarray(table).astype('<u4')
    assert table[-1] <= len(counts)
    h = bytearray(128)
    struct.pack_into('<4sHHIIIf', h, 0, magic, version, 128, frames, len(counts), rate_mhz, lsb)
    struct.pack_into('<2H4H2I', h, 24, *raster, *active, 25000, 1000)
    struct.pack_into('<IIIi3bBII4I', h, 44, readout_us, exposure_us, focal_um, mark_delay_us, *axes, flags, preroll, pitch_nm, 4, 0, 12, 0)
    Path(path).write_bytes(bytes(h) + table.tobytes() + counts.astype('<i2').tobytes())
    return Path(path)


def gyro(path, *opts, points=()):
    args = [str(CLI), 'gyro', str(path), *opts, *[f'pt={x},{y}' for x, y in points]]
    r = subprocess.run(args, capture_output=True, text=True)
    return json.loads(r.stdout)


def frame_sample(i, preroll=600):
    """Sample position of the middle of frame i's readout."""
    return preroll + 100 * i - MARK_DELAY * RATE


def pulse_clip(axis, sign, frames=200, at=100, counts=13100, length=30):
    """Camera still, except that it is turned by counts/131 deg/s * length samples just before
    frame `at` is read and turned back just after: only that frame is displaced."""
    g = np.zeros((600 + 100 * frames, 3), np.int16)
    mid = frame_sample(at)
    a = int(mid - 31 - 4 - length)           # ends before the frame's first row
    b = int(np.ceil(mid + 31 + 4))           # starts after its last row
    g[a:a + length, axis] = sign * counts
    g[b:b + length, axis] = -sign * counts
    return g, counts / 131.0 * length / RATE  # degrees


def wobble_clip(frames=200):
    """Steady shake: 2 Hz pan and 3.1 Hz tilt of about 1.2 and 0.6 degrees."""
    t = np.arange(600 + 100 * frames) / RATE
    g = np.zeros((len(t), 3), np.int16)
    g[:, 1] = np.round(2000 * np.sin(2 * np.pi * 2.0 * t))
    g[:, 0] = np.round(1500 * np.sin(2 * np.pi * 3.1 * t))
    return g


def rough_clip(frames=400):
    """A calm hand-held take (0.7 Hz sway of about 1 degree) with two rough seconds, frames 150..199 (3 Hz, about 5.7 degrees)."""
    t = np.arange(600 + 100 * frames) / RATE
    g = np.zeros((len(t), 3))
    g[:, 1] = 600 * np.sin(2 * np.pi * 0.7 * t)
    g[:, 0] = 400 * np.sin(2 * np.pi * 0.9 * t + 1)
    a, b = int(frame_sample(150)), int(frame_sample(200))
    env = np.zeros(len(t))
    env[a:b] = np.hanning(b - a)
    g[:, 1] += 14000 * env * np.sin(2 * np.pi * 3.0 * t)
    g[:, 0] += 8000 * env * np.sin(2 * np.pi * 2.3 * t)
    return np.round(g).astype(np.int16)


def truncate_fpg(fpg, out, frames):
    """The first `frames` frames of a .FPG file as a file of its own."""
    d = Path(fpg).read_bytes()
    fc = struct.unpack_from('<I', d, 8)[0]
    table = np.frombuffer(d, '<u4', fc, 128)[:frames]
    n = int(table[-1])
    h = bytearray(d[:128])
    struct.pack_into('<II', h, 8, frames, n)
    Path(out).write_bytes(bytes(h) + table.astype('<u4').tobytes() + d[128 + 4 * fc:128 + 4 * fc + 6 * n])
    return Path(out)


def smoothing_share(sigma, fps=25.0, frames=200, at=100):
    """Share of a one-frame displacement that the Gaussian path smoothing keeps."""
    k = np.arange(frames) - at
    u = k / fps / sigma
    w = np.exp(-0.5 * u * u)[np.abs(u) <= 3.0]
    return 1.0 / w.sum()


def project(px, py, rot, f=F_PX, zoom=1.0):
    """Stabilised raster position -> recorded position for a rotation matrix real<-virtual."""
    v = rot @ np.array([(px - CX) / zoom, (py - CY) / zoom, f])
    return CX + f * v[0] / v[2], CY + f * v[1] / v[2]


def rot_axis(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return {'x': np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            'y': np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            'z': np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def quat_path(counts, lsb, rate, axes):
    """Independent integration of the file's samples: orientation at every sample (w, x, y, z)."""
    g = np.radians(counts.astype(np.float64) / lsb)
    v = np.stack([np.sign(a) * g[:, abs(a) - 1] for a in axes], 1)       # pan, tilt, roll
    w = np.stack([-v[:, 1], v[:, 0], v[:, 2]], 1)                        # about x (right), y (down), z (forward)
    step = 0.5 * (w[1:] + w[:-1]) / rate
    q = np.zeros((len(g), 4))
    q[0] = (1, 0, 0, 0)
    for i, r in enumerate(step):
        a = np.linalg.norm(r)
        d = np.array([1.0, *(r / 2)]) if a < 1e-12 else np.array([np.cos(a / 2), *(r * np.sin(a / 2) / a)])
        p = q[i]
        q[i + 1] = (p[0] * d[0] - p[1] * d[1] - p[2] * d[2] - p[3] * d[3],
                    p[0] * d[1] + p[1] * d[0] + p[2] * d[3] - p[3] * d[2],
                    p[0] * d[2] - p[1] * d[3] + p[2] * d[0] + p[3] * d[1],
                    p[0] * d[3] + p[1] * d[2] - p[2] * d[1] + p[3] * d[0])
        q[i + 1] /= np.linalg.norm(q[i + 1])
    return q


def phase_shift(a, b):
    """(dx, dy, peak) with b(x) ~ a(x - d): the picture of a moved by d to give b."""
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    h, w = a.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    r = np.conj(np.fft.rfft2(a * win)) * np.fft.rfft2(b * win)
    c = np.fft.irfft2(r / (np.abs(r) + 1e-12), s=a.shape)
    py, px = np.unravel_index(np.argmax(c), c.shape)

    def sub(m, o, p):
        den = m - 2 * o + p
        return 0.0 if abs(den) < 1e-12 else 0.5 * (m - p) / den
    dy = py + sub(c[(py - 1) % h, px], c[py, px], c[(py + 1) % h, px])
    dx = px + sub(c[py, (px - 1) % w], c[py, px], c[py, (px + 1) % w])
    return (dx - w if dx > w / 2 else dx), (dy - h if dy > h / 2 else dy), float(c[py, px])


# ---- tests ----

def main():
    report = []

    def ok(name, **info):
        report.append(dict(name=name, passed=True, **info))
        print('PASS', name, info if info else '')

    def skip(name, why):
        report.append(dict(name=name, skipped=why))
        print('SKIP', name, '-', why)

    tmp = Path(tempfile.mkdtemp(prefix='sfp_gyro_'))
    static = ['autozoom=0', 'smooth=0.5']

    # -- 1a. parser on the real sample, against an independent read of the same bytes --
    d = SAMPLE.read_bytes()
    fc, sc = struct.unpack_from('<II', d, 8)
    table = np.frombuffer(d, '<u4', fc, 128)
    counts = np.frombuffer(d, '<i2', sc * 3, 128 + 4 * fc).reshape(-1, 3)
    j = gyro(SAMPLE, 'path=1', 'frame=100')
    assert (j['version'], j['frames'], j['samples']) == (1, 292, 29687) == (1, fc, sc), j
    assert abs(j['rate_hz'] - 2499.46) < 1e-6 and j['lsb_per_dps'] == 131.0
    assert j['raster'] == [3264, 2170] and j['active'] == [12, 5, 3240, 2160] and j['fps'] == 25.0
    assert (j['readout_us'], j['exposure_us'], j['focal_um'], j['mark_delay_us']) == (24800, 10000, 28000, 17000)
    assert j['axes'] == [2, -1, 3] and j['flags'] == 1 and j['preroll'] == 600 and j['pitch_nm'] == 0
    assert (j['resolution'], j['dc_crop'], j['bit_depth'], j['sensor_mode']) == (4, 0, 12, 0)
    assert len(d) == 128 + 4 * fc + 6 * sc
    path = np.array(j['path'])
    # Frame time = the mark on the fitted cadence - (readout/2 + exposure/2) = 12.4 + 5.0 ms for this file.
    assert np.abs(path[:, 11] - table / 2499.46).max() < 1e-8 and j['delay_from_exposure'] == 1 and abs(j['delay_ms'] - 17.4) < 1e-6
    lateness = path[:, 11] - path[:, 12]
    assert abs(np.median(lateness)) < 1e-4 and lateness.min() > -0.0006 and lateness.max() < 0.004 and np.abs(np.diff(path[:, 12]) - 0.04).max() < 2e-6
    times = path[:, 12] - 0.0174
    assert np.abs(path[:, 0] - times).max() < 1e-8 and abs(j['time'] - times[100]) < 1e-8
    assert '| sync 17.4 ms = readout/2 + exposure/2 (1/100 s) |' in j['status'] and j['dropped_frames'] == 0 and j['late_marks'] == 0
    q = quat_path(counts, 131.0, 2499.46, (2, -1, 3))
    pos = times * 2499.46
    i0 = np.floor(pos).astype(int)
    qi = q[i0] + (pos - i0)[:, None] * (q[i0 + 1] - q[i0])
    qi /= np.linalg.norm(qi, axis=1)[:, None]
    err = float(np.abs(path[:, 1:5] - qi).max())
    assert err < 1e-9, err
    assert 'On | A001_092.FPG: 292 frames, 2499.46 Hz' in j['status'] and 'gyro data ends at frame 292 (before the end of the clip)' in j['status']
    fallback = 28.0 / (SENSOR_PITCH_MM * 6000 / 3240)          # the active width of this raster
    assert abs(j['focal_px'] - fallback) < 1e-3
    total = np.degrees(2 * np.arccos(np.clip(np.abs((q[:1] * q[table - 1]).sum(1)), 0, 1)))
    ok('sample A001_092.FPG: header, frame times and integrated orientation equal the reference',
       frames=fc, samples=sc, rate_hz=j['rate_hz'], max_quaternion_error=err, focal_px=j['focal_px'],
       largest_rotation_from_start_deg=float(total.max()))

    # -- 1b. malformed files are refused --
    cases = {'truncated': d[:-1000], 'magic': b'XXXX' + d[4:], 'version': d[:4] + struct.pack('<H', 2) + d[6:],
             'short': d[:64], 'axes': d[:60] + bytes([2, 2, 3]) + d[63:]}
    for name, data in cases.items():
        (tmp / f'bad_{name}.FPG').write_bytes(data)
        r = gyro(tmp / f'bad_{name}.FPG')
        assert 'error' in r and 'frames' not in r, (name, r)
    assert 'error' in gyro(tmp / 'missing.FPG')
    ok('truncated, wrong magic, wrong version, short and bad-axis files are refused', cases=len(cases) + 1)

    # -- 1c. a known rotation gives the expected shift --
    share = smoothing_share(0.5)
    shifts = {}
    for label, axis, sign, rot_ax, rot_sign in (('pan', 1, +1, 'y', -1), ('tilt', 0, -1, 'x', +1), ('roll', 2, +1, 'z', -1)):
        g, deg = pulse_clip(axis, sign)
        f = write_fpg(tmp / f'{label}.FPG', g, 200)
        rel = deg * (1 - share)                      # frame 100 against the smoothed camera
        pts = [(CX, CY), (CX + 1000, CY), (400, 300), (2900, 1900)]
        r = gyro(f, 'frame=100', *static, points=pts)
        assert r['on'] == 1 and abs(r['focal_px'] - F_PX) < 1e-6 and r['zoom'] == 1.0
        rot = rot_axis(rot_ax, rot_sign * rel)       # real <- virtual
        worst = 0.0
        for (x, y, sx, sy) in r['points']:
            ex, ey = project(x, y, rot)
            worst = max(worst, abs(sx - ex), abs(sy - ey))
        assert worst < 0.02, (label, worst, r['points'])
        shifts[label] = (r['points'][0][2] - CX, r['points'][0][3] - CY, r['points'][1][2] - CX - 1000, r['points'][1][3] - CY)
        # The neighbouring frames only carry the small smoothing share, with the other sign.
        n = gyro(f, 'frame=90', *static, points=[(CX, CY)])
        assert abs(np.hypot(n['points'][0][2] - CX, n['points'][0][3] - CY)) < F_PX * np.tan(np.radians(deg * share)) + 0.02
    d_px = F_PX * np.tan(np.radians(1.2 * (1 - share)))
    assert abs(shifts['pan'][0] + d_px) < 0.02 and abs(shifts['pan'][1]) < 0.02          # aim right: read further left
    assert abs(shifts['tilt'][1] + d_px) < 0.02 and abs(shifts['tilt'][0]) < 0.02        # aim down: read further up
    assert abs(shifts['roll'][0]) < 0.02 and abs(shifts['roll'][3] + 1000 * np.sin(np.radians(1.2 * (1 - share)))) < 0.03
    ok('known rotation -> expected shift (pan +Y, tilt -X, roll +Z; 1.2 deg at 2800 px)',
       expected_px=float(d_px), pan_dx=shifts['pan'][0], tilt_dy=shifts['tilt'][1], roll_dy_at_1000px=shifts['roll'][3],
       smoothing_share=float(share))

    # -- 1d. the axis map of the header is honoured --
    g, deg = pulse_clip(0, -1)                         # -X
    f = write_fpg(tmp / 'axes.FPG', g, 200, axes=(-1, 3, 2))
    r = gyro(f, 'frame=100', *static, points=[(CX, CY)])
    assert abs(r['points'][0][2] - CX + d_px) < 0.02 and abs(r['points'][0][3] - CY) < 0.02
    ok('axis map (-1, 3, 2): -X is read as pan')

    # -- 1e. focal length: pitch from the file, fallback pitch, unknown lens, override --
    g, _ = pulse_clip(1, 1)
    f0 = write_fpg(tmp / 'nofocal.FPG', g, 200, focal_um=0)
    r = gyro(f0, 'frame=100')
    assert r['on'] == 0 and 'focal length' in r['why_off'] and r['status'].startswith('Off: ')
    r = gyro(f0, 'frame=100', 'focal=35')
    assert r['on'] == 1 and abs(r['focal_px'] - 3500) < 1e-6 and '35.0 mm (override)' in r['status']
    r = gyro(write_fpg(tmp / 'nopitch.FPG', g, 200, pitch_nm=0), 'frame=100')
    assert abs(r['focal_px'] - fallback) < 1e-3
    r = gyro(tmp / 'pan.FPG', 'frame=100', 'stab=0')
    assert r['on'] == 0 and r['status'].startswith('Off (switched off)')
    ok('focal length: file pitch, whole sensor width when no pitch, manual lens needs the override, switch',
       fallback_px_at_28mm=float(fallback))

    # -- 1f. rolling shutter: constant pan, virtual camera = camera at the readout middle --
    g = np.zeros((600 + 100 * 120, 3), np.int16)
    g[:, 1] = 13100                                    # 100 deg/s
    f = write_fpg(tmp / 'rs.FPG', g, 120)
    pts = [(CX, 0), (CX, CY), (CX, RASTER[1]), (300, 200), (3000, 2000), (1000, 1800)]
    rs_info = {}
    for amount in (1.0, 0.5, 0.0):
        r = gyro(f, 'frame=60', 'smooth=0', 'autozoom=0', f'rs={amount}', points=pts)
        worst = 0.0
        for (x, y, sx, sy) in r['points']:
            ey = y
            for _ in range(30):                        # the row that was read at the matching time
                theta = 100.0 * (ey / RASTER[1] - 0.5) * READOUT * amount
                ex, ey = project(x, y, rot_axis('y', -theta))
            worst = max(worst, abs(sx - ex), abs(sy - ey))
        assert worst < 0.05, (amount, worst, r['points'])
        rs_info[f'top_row_dx_rs{amount}'] = r['points'][0][2] - CX
    top = F_PX * np.tan(np.radians(100.0 * 0.5 * READOUT))
    assert abs(rs_info['top_row_dx_rs1.0'] - top) < 0.05 and abs(rs_info['top_row_dx_rs0.0']) < 0.01
    # The readout time can be overridden (the plug-in's "Rolling Shutter ms").
    r = gyro(f, 'frame=60', 'smooth=0', 'autozoom=0', 'readout=12.4', points=[(CX, 0)])
    half = F_PX * np.tan(np.radians(100.0 * 0.5 * 0.0124))
    assert abs(r['points'][0][2] - CX - half) < 0.05 and 'readout 12.4 ms (override)' in r['status'], r['points']
    rs_info['top_row_dx_readout_12.4ms'] = r['points'][0][2] - CX
    # Zero-phase smoothing leaves a steady pan alone.
    r = gyro(f, 'frame=60', 'smooth=0.5', 'autozoom=0', points=[(CX, CY)])
    assert np.hypot(r['points'][0][2] - CX, r['points'][0][3] - CY) < 0.01
    ok('rolling shutter model: 100 deg/s pan, 24.8 ms readout (first row earlier, last row later)',
       expected_top_row_dx=float(top), **rs_info)

    # -- 1g. sync offset is added to the mark delay --
    a, b = gyro(f, 'frame=60'), gyro(f, 'frame=60', 'sync=5')
    assert abs(a['time'] - (frame_sample(60) / RATE)) < 1e-8 and abs(a['time'] - b['time'] - 0.005) < 1e-8
    turn = np.degrees(2 * np.arccos(min(1.0, abs(np.dot(a['orientation'], b['orientation'])))))
    assert abs(turn - 0.5) < 1e-4, turn
    ok('sync offset: +5 ms reads the gyro 5 ms earlier', frame_time=a['time'], rotation_between_deg=float(turn))

    # -- 1g2. frame times from a regular cadence: late marks, drift, dropped frames --
    rng = np.random.default_rng(5)
    nf = 1500
    calm = np.zeros((600 + 101 * nf + 400, 3), np.int16)

    def marks_of(table, **kw):
        r = gyro(write_fpg(tmp / 'marks.FPG', calm, nf, table=table, **kw), 'path=1', 'autozoom=0')
        return np.array(r['path'])[:, 12] * RATE, r
    jitter = rng.integers(0, 3, nf)                           # every mark up to 0.8 ms late, as on the camera
    true = 600 + 100.0 * np.arange(nf)
    late_at = {200: 52, 201: 8, 640: 93, 641: 44, 642: 5, 900: 139, 901: 90, 902: 41, 1499: 60, 0: 30}
    table = true + jitter
    for k, v in late_at.items():
        table[k] = true[k] + v
    got, r = marks_of(np.maximum.accumulate(table))
    err = got - (true + 1)                                    # an undisturbed mark is 1 sample late on average here
    assert np.abs(err).max() < 1.0 and r['dropped_frames'] == 0 and r['late_marks'] == sum(1 for v in late_at.values() if v > 15), (np.abs(err).max(), r['late_marks'])
    assert f"| {r['late_marks']} late frame marks corrected" in r['status']
    raw_err = np.array(gyro(tmp / 'marks.FPG', 'path=1', 'autozoom=0', 'marks=raw')['path'])[:, 0] * RATE
    info = dict(late_marks=r['late_marks'], worst_late_ms=139 / 2.5, largest_error_ms_regular=float(np.abs(err).max() / 2.5),
                typical_lateness_ms=r['typical_mark_lateness_ms'])
    # Drift: the frame period is 0.05 % longer than nominal and wanders; the cadence follows.
    period = 100.05 + 0.02 * np.sin(np.arange(nf) / 150.0)
    true_d = 600 + np.concatenate([[0], np.cumsum(period[1:])])
    got, r = marks_of(np.floor(true_d + jitter))
    err = got - (true_d + 0.5)
    assert np.abs(err).max() < 1.2 and r['dropped_frames'] == 0 and true_d[-1] - true[-1] > 70, np.abs(err).max()
    info['drift_total_ms'] = float((true_d[-1] - true[-1]) / 2.5)
    info['largest_error_ms_with_drift'] = float(np.abs(err).max() / 2.5)
    # Dropped frames: the cadence went on while no frame was recorded. That gap stays.
    true_g = true.copy()
    true_g[300:] += 100                                       # one frame dropped before frame 300
    true_g[1000:] += 200                                      # two before frame 1000
    table = true_g + jitter
    table[700] += 60                                          # and a late mark somewhere else
    table[299] += 45                                          # also right before the drop
    got, r = marks_of(np.maximum.accumulate(table))
    err = got - (true_g + 1)
    assert r['dropped_frames'] == 3 and np.abs(err).max() < 1.0 and '| 3 frames dropped by the camera' in r['status'], (r['dropped_frames'], np.abs(err).max())
    assert abs((got[300] - got[299]) - 200) < 1.0 and abs((got[1000] - got[999]) - 300) < 1.0
    info['dropped_frames_found'] = r['dropped_frames']
    ok('frame times on the regular cadence: late marks removed, drift followed, dropped frames kept', **info)

    # -- 1g3. sync from the exposure: readout/2 + exposure/2; without an exposure time the file's value --
    g0 = np.zeros((600 + 100 * 50, 3), np.int16)
    a = gyro(write_fpg(tmp / 'exp.FPG', g0, 50, exposure_us=1563), 'frame=10')
    assert a['delay_from_exposure'] == 1 and abs(a['delay_ms'] - (12.4 + 0.7815)) < 1e-6 and '| sync 13.2 ms = readout/2 + exposure/2 (1/640 s) |' in a['status'], a['status']
    assert abs(a['time'] - (frame_sample(10) / RATE + 0.017 - a['delay_ms'] / 1e3)) < 1e-8
    b = gyro(write_fpg(tmp / 'noexp.FPG', g0, 50), 'frame=10', 'sync=-2')
    assert b['delay_from_exposure'] == 0 and b['delay_ms'] == 17.0 and '| sync 17.0 ms from the gyro data (exposure time unknown) -2.0 ms offset |' in b['status'], b['status']
    c = gyro(write_fpg(tmp / 'noro.FPG', g0, 50, exposure_us=10000, readout_us=0), 'frame=10')
    assert c['delay_from_exposure'] == 0 and c['delay_ms'] == 17.0 and '(readout time unknown)' in c['status']
    # Image scale: every mode before R122 covers the whole 36 mm sensor width (framing test of 2026-10-02).
    mq = gyro(write_fpg(tmp / 'mq.FPG', g0, 50, raster=(3024, 2010), active=(8, 5, 3008, 2000), pitch_nm=0))
    hq = gyro(write_fpg(tmp / 'hq.FPG', g0, 50, pitch_nm=0))
    assert abs(mq['focal_px'] - 28 / (0.006 * 6000 / 3008)) < 1e-3 and abs(hq['focal_px'] - 28 / (0.006 * 6000 / 3240)) < 1e-3
    assert mq['window'] == [6000, 4000] and mq['recorded'] == [3008, 2000] and 'sensor window' not in mq['status']
    ok('sync = readout/2 + exposure/2 when the exposure time is known, else the value in the gyro data; image scale per mode',
       delay_ms_at_1_640s=a['delay_ms'], delay_ms_exposure_unknown=b['delay_ms'], focal_px_mq=mq['focal_px'], focal_px_hq=hq['focal_px'])

    # -- 1h. auto zoom hides the borders, and no more --
    r = gyro(tmp / 'pan.FPG', 'frame=100', 'smooth=0.5')
    zoom = r['auto_zoom']
    x0, y0, x1, y1 = ACTIVE[0], ACTIVE[1], ACTIVE[0] + ACTIVE[2], ACTIVE[1] + ACTIVE[3]
    border = [(x, y) for x in np.linspace(x0, x1, 9) for y in (y0, y1)] + [(x, y) for y in np.linspace(y0, y1, 9) for x in (x0, x1)]

    def outside(res):
        return sum(1 for (_, _, sx, sy) in res['points'] if sx < x0 - 0.01 or sx > x1 + 0.01 or sy < y0 - 0.01 or sy > y1 + 0.01)
    assert 1.03 < zoom < 1.07 and r['zoom'] == zoom
    assert outside(gyro(tmp / 'pan.FPG', 'frame=100', 'smooth=0.5', points=border)) == 0
    assert outside(gyro(tmp / 'pan.FPG', 'frame=100', 'smooth=0.5', 'autozoom=0', f'zoom={zoom * 0.995}', points=border)) > 0
    r2 = gyro(tmp / 'pan.FPG', 'frame=100', 'smooth=0.5', 'zoom=1.5')
    assert abs(r2['zoom'] - 1.5 * zoom) < 1e-5
    half = ACTIVE[2] / 2 / F_PX                                         # the edge that the 1.16 degree turn pushes out
    expected_zoom = half / np.tan(np.arctan(half) - np.radians(1.2 * (1 - share)))
    assert abs(zoom - expected_zoom) < 0.002, (zoom, expected_zoom)
    ok('auto zoom: borders hidden on every frame, 0.5 % less shows a border; manual zoom multiplies',
       auto_zoom=zoom, expected=float(expected_zoom))

    # -- 1h2. zoom limit: frames that need more are stabilised less, and still show no border --
    r = gyro(tmp / 'pan.FPG', 'frame=100', 'smooth=0.5', 'maxzoom=1.02', points=[(CX, CY)])
    assert r['zoom'] == 1.02 and r['auto_zoom'] == 1.02 and r['limited_frames'] > 0 and 'at the limit' in r['status']
    reduced = CX - r['points'][0][2]
    assert 5 < reduced < 0.6 * d_px, reduced
    for fr in range(85, 116):
        assert outside(gyro(tmp / 'pan.FPG', f'frame={fr}', 'smooth=0.5', 'maxzoom=1.02', points=border)) == 0, fr
    far = gyro(tmp / 'pan.FPG', 'frame=60', 'smooth=0.5', 'maxzoom=1.02')
    assert far['smoothed'] == gyro(tmp / 'pan.FPG', 'frame=60', 'smooth=0.5', 'autozoom=0')['smoothed']
    assert gyro(tmp / 'pan.FPG', 'frame=100', 'smooth=0.5')['limited_frames'] == 0          # default limit 1.3: untouched
    # The real sample has pans and tilts of +-20 degrees: it sits at the default limit of 1.3.
    r = gyro(SAMPLE, 'frame=0')
    assert r['zoom'] == 1.3 and r['limited_frames'] > 0
    for fr in range(0, 292, 7):
        assert outside(gyro(SAMPLE, f'frame={fr}', points=border)) == 0, fr
    ok('zoom limit: correction reduced where the limit is reached, no border on any checked frame',
       limit=1.02, full_correction_px=float(d_px), reduced_correction_px=reduced, sample_frames_reduced_at_1_3=r['limited_frames'])

    # -- 1h3. used range: smoothing and zoom are worked out for that part only --
    rough = write_fpg(tmp / 'rough.FPG', rough_clip(), 400)
    opts = ['clip=400', 'maxzoom=3']
    whole = gyro(rough, 'path=1', *opts)
    part = gyro(rough, 'path=1', 'range=1-140', *opts)
    alone = gyro(truncate_fpg(rough, tmp / 'rough140.FPG', 140), 'path=1', 'clip=140', 'maxzoom=3')
    assert whole['range'] == [1, 400] and whole['status'].endswith('| range: whole clip')
    assert part['range'] == [1, 140] and part['status'].endswith('| range: frames 1 to 140 of 400 (manual)'), part['status']
    assert whole['zoom'] > 1.25 and part['zoom'] < 1.08 and part['zoom'] <= alone['zoom'] + 1e-6, (whole['zoom'], part['zoom'], alone['zoom'])
    pp, pa_ = np.array(part['path']), np.array(alone['path'])
    if part['smoothing_past_range'] == [0, 0]:                         # the range smoothed as a clip of its own
        assert np.abs(pp[:140, 5:9] - pa_[:, 5:9]).max() < 1e-9 and abs(part['zoom'] - alone['zoom']) < 1e-6
    assert np.abs(pp[:140, 5:9] - np.array(whole['path'])[:140, 5:9]).max() > 1e-4
    for fr in list(range(0, 140, 9)) + [139]:                          # no border inside the range at its zoom
        assert outside(gyro(rough, f'frame={fr}', 'range=1-140', *opts, points=border)) == 0, fr
    cut = {}
    for fr in (139, 140, 150, 165, 180, 199, 230, 399):                # outside it: same zoom, less correction, still no border
        r = gyro(rough, f'frame={fr}', 'range=1-140', *opts, points=[(CX, CY)] + border)
        assert r['zoom'] == part['zoom'] and outside({'points': r['points'][1:]}) == 0, fr
        full = gyro(rough, f'frame={fr}', *opts, points=[(CX, CY)])
        cut[fr] = (np.hypot(r['points'][0][2] - CX, r['points'][0][3] - CY), np.hypot(full['points'][0][2] - CX, full['points'][0][3] - CY))
    assert cut[165][0] < 0.5 * cut[165][1] and cut[180][0] < 0.5 * cut[180][1], cut
    later = gyro(rough, 'range=230-400', *opts)
    assert later['range'] == [230, 400] and later['zoom'] < 1.08
    assert gyro(rough, 'range=350-9999', *opts)['range'] == [350, 400] and gyro(rough, 'range=300-200', *opts)['range'] == [1, 400]
    assert abs(gyro(rough, 'range=100-280', *opts)['zoom'] - whole['zoom']) < 0.02   # a range with the rough part costs the same
    # A cut in the middle of a steady pan: the smoothing runs on past the cut, so the path does not lag at the ends.
    pan = np.zeros((600 + 100 * 200, 3), np.int16)
    pan[:, 1] = 1310                                                   # 10 deg/s
    steady = write_fpg(tmp / 'steady.FPG', pan, 200)
    mid = gyro(steady, 'range=60-140', 'clip=200')
    own = gyro(truncate_fpg(steady, tmp / 'steady80.FPG', 80), 'clip=80')
    assert mid['smoothing_past_range'] == [1, 1] and mid['zoom'] < 1.005 and own['zoom'] > 1.05, (mid['zoom'], own['zoom'])
    ok('used range: zoom and smoothing for that part only; the cut ends are smoothed whichever way needs less zoom; frames outside show no border',
       smoothing_past_cut_rough_clip=part['smoothing_past_range'], steady_pan_zoom_with_smoothing_past_the_cut=mid['zoom'],
       steady_pan_zoom_if_the_cut_were_a_clip_end=own['zoom'],
       zoom_whole_clip=whole['zoom'], zoom_frames_1_to_140=part['zoom'], zoom_of_that_part_as_its_own_clip=alone['zoom'],
       zoom_frames_230_to_400=later['zoom'], correction_px_inside_rough_part_with_and_without_range={k: [float(a), float(b)] for k, (a, b) in cut.items()})

    # -- 1h4. dynamic zoom: follows the local need, slowly, never below it --
    info = {}
    for sm in (2.0, 4.0):
        d = gyro(rough, 'path=1', 'zoommode=dynamic', f'zoomsmooth={sm}', *opts)
        z = np.array(d['path'])[:, 9]
        w = sm * 25
        step = float(np.abs(np.diff(z)).max())
        assert abs(z.max() - whole['zoom']) < 2e-3 and z.min() < 1.08 and d['zoom_max'] == float(z.max()) and abs(d['zoom_mean'] - z.mean()) < 1e-5
        assert z[:int(150 - 2 * w)].max() < 1.08 if 150 - 2 * w > 10 else True
        assert step < 1.3 * (z.max() - z.min()) / w, (sm, step)        # eased over the look-ahead time
        assert np.abs(np.diff(z, 2)).max() < 0.25 * step               # no kink either
        for fr in list(range(0, 400, 7)) + list(range(140, 215)):      # never below the need: no border on any frame
            assert outside(gyro(rough, f'frame={fr}', 'zoommode=dynamic', f'zoomsmooth={sm}', *opts, points=border)) == 0, (sm, fr)
        assert f'zoom {z.min():.3f} to {z.max():.3f} (dynamic, mean {z.mean():.3f})' in d['status'], d['status']
        info[f'smoothness_{sm:g}s'] = dict(min=float(z.min()), mean=float(z.mean()), max=float(z.max()), largest_change_per_second=step * 25)
    fixed_virtual = np.array(whole['path'])[:, 5:9]
    assert np.abs(np.array(d['path'])[:, 5:9] - fixed_virtual).max() < 1e-9          # same stabilisation, only the zoom differs
    dr = gyro(rough, 'path=1', 'zoommode=dynamic', 'range=1-140', *opts)
    assert dr['zoom_max'] <= part['zoom'] + 2e-3 and np.array(dr['path'])[150:, 9].max() <= dr['zoom_max'] + 1e-6
    lim = gyro(rough, 'path=1', 'zoommode=dynamic')                    # default limit 1.3: capped there
    assert abs(lim['zoom_max'] - 1.3) < 1e-6 and lim['limited_frames'] == gyro(rough, 'clip=400')['limited_frames'] > 0
    for fr in range(140, 215, 3):
        assert outside(gyro(rough, f'frame={fr}', 'zoommode=dynamic', points=border)) == 0, fr
    ok('dynamic zoom: reaches the fixed zoom only near the rough part, changes slowly and smoothly, no border on any frame',
       fixed_zoom=whole['zoom'], **info)

    # -- 1i. frames after the end of the gyro data are not rotated --
    r = gyro(tmp / 'pan.FPG', 'frame=5000', *static, points=[(100, 100)])
    assert r['on'] == 0 and r['points'][0][2:] == [100, 100]
    r = gyro(tmp / 'pan.FPG', 'frame=5000', *static, 'zoom=1.25', points=[(CX + 100, CY)])
    assert r['on'] == 1 and r['zoom'] == 1.25 and abs(r['points'][0][2] - (CX + 80)) < 1e-3 and abs(r['points'][0][3] - CY) < 1e-3
    ok('frames beyond the file: not warped (manual zoom only)')

    # -- 1k. gyro data that ends before the clip: correction and zoom fade out over the last second --
    g = wobble_clip()
    fade = write_fpg(tmp / 'fade.FPG', g, 200, flags=1)
    j = gyro(fade, 'path=1', 'frame=199', points=[(CX, CY), (300, 300), (3000, 1900)])
    assert j['ends_early'] == 1 and 'gyro data ends at frame 200 (before the end of the clip)' in j['status']
    assert 'gyro data ends at frame 200 of 500:' in gyro(fade, 'clip=500')['status']
    whole = gyro(fade, 'path=1', 'clip=200')                 # as many DNGs as gyro frames: nothing to fade
    assert whole['ends_early'] == 0 and 'ends at' not in whole['status']
    unflagged = gyro(write_fpg(tmp / 'nofade.FPG', g, 200, flags=0), 'path=1')
    assert unflagged['ends_early'] == 0 and unflagged['path'] == whole['path']
    assert gyro(tmp / 'nofade.FPG', 'clip=500')['ends_early'] == 1         # more DNGs than gyro frames
    zf, zw = np.array(j['path'])[:, 9], np.array(whole['path'])[:, 9]
    assert j['zoom'] == 1.0 and j['frame_zoom'] == 1.0 and zf[199] == 1.0
    assert max(abs(a - x) + abs(b - y) for x, y, a, b in j['points']) < 0.01          # last gyro frame: as recorded
    assert np.all(zw == zw[0]) and zw[0] > 1.03 and np.array_equal(zf[:175], zw[:175])
    assert np.all(np.diff(zf[174:]) <= 0) and np.abs(np.diff(zf)).max() < 0.1 * (zw[0] - 1) + 1e-6
    assert j['path'][174] == whole['path'][174] and j['path'][100] == whole['path'][100]

    def centre(file, fr, *o):
        r = gyro(file, f'frame={fr}', *o, points=[(CX, CY)] + border)
        return np.array(r['points'][0][2:]) - (CX, CY), outside({'points': r['points'][1:]})
    cf = [centre(fade, fr) for fr in range(170, 200)]
    cw = [centre(fade, fr, 'clip=200') for fr in range(170, 200)]
    assert all(o == 0 for _, o in cf)                                               # no border inside the fade
    mf, mw = np.array([np.hypot(*c) for c, _ in cf]), np.array([np.hypot(*c) for c, _ in cw])
    assert np.all(mf <= mw + 0.01) and mw[-1] > 5 and mf[-1] < 0.01
    step_f = np.abs(np.diff(np.array([c for c, _ in cf]), axis=0)).max()
    step_w = np.abs(np.diff(np.array([c for c, _ in cw]), axis=0)).max()
    assert step_f <= step_w + 0.05, (step_f, step_w)
    ok('end of gyro data: zoom and correction fade to none over the last second, no border, no step',
       auto_zoom=float(zw[0]), zoom_last_frames=[float(z) for z in zf[-5:]], correction_px_last_frame_without_fade=float(mw[-1]),
       correction_px_last_frame=float(mf[-1]), largest_frame_to_frame_change_px=float(step_f), same_without_fade=float(step_w))

    # -- 1j. sidecar lookup by clip name --
    look = tmp / 'A001_092'
    look.mkdir()
    dng = look / 'A001_092_20261001_000001.DNG'
    assert gyro(dng)['error'] == 'no gyro data in the DNGs, no .FPG gyro file next to the DNGs' 
    shutil.copy(SAMPLE, look / 'B777_001.FPG')
    assert Path(gyro(dng)['file']).name == 'B777_001.FPG'               # the only one in the folder
    shutil.copy(SAMPLE, look / 'C123_456.FPG')
    assert 'none named like the clip' in gyro(dng)['error']
    look2 = tmp / 'reel'
    look2.mkdir()
    shutil.copy(SAMPLE, look2 / 'reel.fpg')
    shutil.copy(SAMPLE, look2 / 'other.FPG')
    assert Path(gyro(look2 / 'X_000001.DNG')['file']).name.lower() == 'reel.fpg'   # named like the folder
    shutil.copy(SAMPLE, look / 'A001_092.FPG')
    shutil.copy(SAMPLE, look / 'A001.FPG')
    assert Path(gyro(dng)['file']).name == 'A001_092.FPG'               # longest clip-name match
    ok('sidecar lookup: clip name, folder name, only file; none / ambiguous -> off')

    # ---- 2. the plug-in on the GPU ----
    frames = sorted(CLIP.glob('*.DNG')) if CLIP.is_dir() else []
    if len(frames) < 291:
        skip('plug-in and real-clip tests', f'{CLIP} is not connected')
    else:
        render_paths(frames, tmp, ok)
        plugin_tests(frames, tmp, ok, d_px, share)
        real_clip(frames, ok)
        for name in ('A001_092', 'A001_093', 'A001_094'):
            rolling_shutter_check(CLIP.parent / name, ok, skip)
        end_of_data(CLIP.parent / 'A001_093', ok, skip)
    mq_clips(ok, skip)

    shutil.rmtree(tmp, ignore_errors=True)
    (ROOT / 'tests/gyro_report.json').write_text(json.dumps({'plugin': str(PLUGIN), 'clip': str(CLIP), 'tests': report}, indent=2) + '\n')
    done = sum(1 for r in report if r.get('passed'))
    print(f'ALL PASS ({done})' + (f', {len(report) - done} skipped' if done != len(report) else ''))


def source(host, first):
    host.obj(host.obj(host.instance)['props'])['values']['OfxImageEffectPropSrcFilePath'] = [str(first)]
    host.changed('Source')


def green(out):
    return out.pixels[::-1, :, 1].astype(np.float64)       # top-down


def render_paths(frames, tmp, ok):
    """The two ways the plug-in renders: into host memory (own CUDA context, download) and into a
    GPU image of the host's context (what Resolve uses in CUDA mode)."""
    import tifffile
    out = {}
    for rs in (0, 1):
        for device in (0, 1):
            f = tmp / f'path_{rs}_{device}.tif'
            subprocess.run([str(CLI), 'develop', str(frames[99]), str(f), f'gyro={SAMPLE}', 'width=1620', 'height=1080', f'rs={rs}',
                            f'device={device}'], check=True, capture_output=True)
            out[rs, device] = tifffile.imread(f)
    assert np.array_equal(out[0, 0], out[0, 1]) and np.array_equal(out[1, 0], out[1, 1])
    changed = float((out[0, 1] != out[1, 1]).any(axis=2).mean())
    assert changed > 0.5, changed
    ok('host-memory and GPU-image render paths give the same picture; rolling shutter 0 and 1 differ in both',
       share_of_pixels_changed_by_rolling_shutter=changed)


def band_shifts(host, count, first=0):
    """Per frame pair: picture shift of a top and a bottom band, raster px, with the correlation peaks."""
    prev, out = None, []
    for i in range(first, first + count):
        st, img = host.render(source_frame=i)
        assert st == OK, host.messages[-2:]
        g = green(img)
        cur = (g[80:460, 405:1215], g[620:1000, 405:1215])
        if prev is not None:
            t, b = phase_shift(prev[0], cur[0]), phase_shift(prev[1], cur[1])
            out.append((2 * (b[0] - t[0]), 2 * (b[1] - t[1]), min(t[2], b[2])))
        prev = cur
    return np.array(out)


def rolling_shutter_check(clip, ok, skip):
    """Direction and amount of the rolling-shutter correction, measured on the picture.
    A turning camera shifts the bottom band against the top band (read 12.4 ms apart for these
    band centres). The gyro predicts that shear per frame pair; the slope of the measured shear
    against the predicted one is 1 for an uncorrected picture, 0 for a corrected one and 2 if
    the correction were applied the wrong way round."""
    frames = sorted(clip.glob('*.DNG')) if clip.is_dir() else []
    fpgs = sorted(clip.glob('*.FPG')) if clip.is_dir() else []
    fpg = fpgs[0] if fpgs else (SAMPLE if clip.name == 'A001_092' else None)
    if len(frames) < 266 or fpg is None:
        return skip(f'{clip.name}: rolling shutter direction and amount', 'clip or gyro file not connected')
    N = 265                                                  # inside the gyro data, before its fade-out
    d = fpg.read_bytes()
    fc, sc, rate_mhz = struct.unpack_from('<III', d, 8)
    rate, readout, delay = rate_mhz / 1000, struct.unpack_from('<I', d, 44)[0] * 1e-6, struct.unpack_from('<i', d, 56)[0] * 1e-6
    table = np.frombuffer(d, '<u4', fc, 128)
    deg = np.cumsum(np.frombuffer(d, '<i2', sc * 3, 128 + 4 * fc).reshape(-1, 3) / 131.0, 0) / rate
    focal = gyro(fpg)['focal_px']
    t = table[:N] / rate - delay

    def moved(axis, sign, row):                              # picture motion of a raster row between frames, px
        a = np.interp((t + (row / RASTER[1] - 0.5) * readout) * rate, np.arange(sc), deg[:, axis])
        return -focal * np.radians(np.diff(sign * a))
    rows = (2 * 270 + ACTIVE[1], 2 * 810 + ACTIVE[1])        # centres of the two bands
    pred = np.stack([moved(1, 1, rows[1]) - moved(1, 1, rows[0]), moved(0, -1, rows[1]) - moved(0, -1, rows[0])], 1)

    res, keep = {}, np.ones(N - 1, bool)
    for name, values in (('unstabilised', dict(stabEnable=0)), ('amount_0', dict(stabRollingShutter=0.0)),
                         ('amount_1', dict(stabRollingShutter=1.0)), ('amount_minus_1', dict(stabRollingShutter=-1.0))):
        host = OFXHost(PLUGIN, canvas=(1620, 1080))
        host.set(**STAB_TEST)
        source(host, frames[0])
        host.set(decodeQuality=1, stabAutoZoom=0, gyroFile='' if fpgs else str(fpg), **values)
        res[name] = band_shifts(host, N)
        host.close()
        keep &= res[name][:, 2] > 0.05
    info = {'pairs': int(keep.sum()), 'predicted_shear_rms_px': [float(np.sqrt((pred[keep, k] ** 2).mean())) for k in (0, 1)]}
    for name, v in res.items():
        info[name] = {'slope_x': float(np.polyfit(pred[keep, 0], v[keep, 0], 1)[0]), 'slope_y': float(np.polyfit(pred[keep, 1], v[keep, 1], 1)[0]),
                      'shear_rms_x': float(np.sqrt((v[keep, 0] ** 2).mean())), 'shear_median_abs_x': float(np.median(np.abs(v[keep, 0])))}
    sx = {k: info[k]['slope_x'] for k in res}
    assert sx['amount_1'] < sx['amount_0'] - 0.5 and sx['amount_minus_1'] > sx['amount_0'] + 0.5, sx
    assert abs(sx['amount_1']) < 0.35 and 0.65 < sx['amount_0'] < 1.45, sx
    ok(f'{clip.name}: rolling shutter - slope of measured against gyro-predicted top/bottom shear (1 = uncorrected, 0 = corrected, 2 = reversed)', **info)


def end_of_data(clip, ok, skip):
    """A001_093: 293 gyro frames, 1045 DNGs. The picture must not jump where the gyro data ends."""
    frames = sorted(clip.glob('*.DNG')) if clip.is_dir() else []
    fpgs = sorted(clip.glob('*.FPG')) if clip.is_dir() else []
    if len(frames) < 320 or not fpgs:
        return skip(f'{clip.name}: end of the gyro data', 'clip or gyro file not connected')
    j = gyro(frames[0], 'path=1')
    n = j['frames']
    assert j['clip_frames'] == len(frames) and f'gyro data ends at frame {n} of {len(frames)}' in j['status']
    zoom = np.array(j['path'])[:, 9]
    first, count = n - 45, 70
    win = (slice(270, 810), slice(405, 1215))

    def run(**values):
        host = OFXHost(PLUGIN, canvas=(1620, 1080))
        host.set(**STAB_TEST)
        source(host, frames[0])
        status = host.get('stabStatus')
        host.set(decodeQuality=1, **values)
        seq = [green(host.render(source_frame=i)[1])[win] for i in range(first, first + count)]
        last = [host.render(source_frame=i)[1].pixels.copy() for i in (n - 1, n, n + 5)]
        host.close()
        return np.array([phase_shift(seq[i - 1], seq[i]) for i in range(1, count)]), last, status
    raw, raw_last, _ = run(stabEnable=0)
    stab, stab_last, status = run()
    assert f'gyro data ends at frame {n} of {len(frames)}' in status and status.startswith(f'On | {fpgs[0].name}'), status
    assert np.array_equal(stab_last[1], raw_last[1]) and np.array_equal(stab_last[2], raw_last[2])      # after the data: untouched
    last_diff = float(np.abs(stab_last[0] - raw_last[0]).max())
    assert last_diff < 0.02, last_diff                                                                 # last gyro frame: as recorded
    z = np.r_[zoom, np.ones(40)][first:first + count]
    zpair = 0.5 * (z[1:] + z[:-1])
    s = stab[:, :2] * 2 / zpair[:, None]
    r = raw[:, :2] * 2
    idx = np.arange(first, first + count - 1)                # pair idx -> idx + 1 (0-based frames)
    gap = np.hypot(*(s - r).T)
    assert gap[idx >= n - 1].max() < 0.5 and gap[idx >= n - 4].max() < 6.0, gap[idx >= n - 6]
    jit = lambda v: np.hypot(*np.diff(v, axis=0).T)
    fading = (idx[1:] >= n - 27) & (idx[1:] < n)
    # What the hard cut gave: the full correction and zoom on the last gyro frame, none on the next.
    old = gyro(frames[0], f'frame={n - 1}', f'clip={n}', points=[(CX, CY)])
    info = dict(gyro_frames=n, clip_frames=len(frames), auto_zoom=j['auto_zoom'], zoom_over_last_frames=[float(v) for v in zoom[-26::5]],
                largest_zoom_change_per_frame=float(np.abs(np.diff(z)).max()),
                jump_before_fix_px=float(np.hypot(old['points'][0][2] - CX, old['points'][0][3] - CY)), zoom_jump_before_fix=old['zoom'],
                stabilised_minus_unstabilised_shift_px=dict(last_three_gyro_pairs=float(gap[(idx >= n - 4) & (idx < n - 1)].max()),
                                                            from_the_last_gyro_frame_on=float(gap[idx >= n - 1].max())),
                jitter_max_last_second_px=dict(unstabilised=float(jit(r)[fading].max()), stabilised=float(jit(s)[fading].max())),
                last_gyro_frame_max_pixel_difference=last_diff)
    assert info['largest_zoom_change_per_frame'] < 0.03
    assert info['jitter_max_last_second_px']['stabilised'] < info['jitter_max_last_second_px']['unstabilised'] + 1.5, info
    ok(f'{clip.name} (camera-written gyro file, found next to the DNGs): fade-out at the end of the gyro data', **info)


def plugin_tests(frames, tmp, ok, d_px, share):
    S = 2.0                                                 # canvas pixel -> raster pixel
    host = OFXHost(PLUGIN, canvas=(1620, 1080))
    names = {'stabStatus', 'stabEnable', 'stabSmoothness', 'stabRollingShutter', 'stabSync', 'stabFocal', 'stabAutoZoom', 'stabMaxZoom',
             'stabZoom', 'gyroFile', 'stabRange', 'stabRangeStart', 'stabRangeEnd', 'stabZoomMode', 'stabZoomSmooth', 'stabReadout',
             'stabFocalAuto', 'stabReadoutAuto'}
    assert names <= set(host.params()), names - set(host.params())
    for n in names:
        p = host.obj(host.obj(host.params()[n])['props'])['values']
        assert p.get('OfxParamPropParent') == ['stab'], n
    defaults = {n: host.get(n) for n in sorted(names)}
    assert host.choices('stabRange') == ['Whole Clip', 'Automatic', 'Manual'] and host.choices('stabZoomMode') == ['Fixed', 'Dynamic']
    assert [defaults.pop(k) for k in ('stabRange', 'stabRangeStart', 'stabRangeEnd', 'stabZoomMode', 'stabZoomSmooth')] == [1, 1.0, 0.0, 1, 4.0]
    assert [defaults.pop(k) for k in ('stabReadout', 'stabFocalAuto', 'stabReadoutAuto')] == [0.0, 1, 1]
    assert defaults == {'gyroFile': '', 'stabAutoZoom': 1, 'stabEnable': 0, 'stabFocal': 0.0, 'stabMaxZoom': 1.3, 'stabRollingShutter': 1.0,
                        'stabSmoothness': 0.1, 'stabStatus': 'Off: no DNG source', 'stabSync': 0.0, 'stabZoom': 1.0}, defaults
    gp = host.obj(host.obj(host.params()['gyroFile'])['props'])['values']
    zp = host.obj(host.obj(host.params()['stabZoomSmooth'])['props'])['values']
    assert gp.get('OfxParamPropSecret') == [1] and zp.get('OfxPropLabel') == ['Smoothness'] and zp.get('OfxParamPropEnabled') == [1]
    host.set(stabZoomMode=0); host.changed('stabZoomMode')
    assert zp.get('OfxParamPropEnabled') == [0]
    host.set(stabZoomMode=1); host.changed('stabZoomMode')
    assert zp.get('OfxParamPropEnabled') == [1]
    ok('Stabilisation group declared, defaults (1.6.0): off, smoothness 0.1 s, rolling shutter 1, sync 0, focal from file, auto zoom up to 1.3, '
       'range automatic, zoom mode dynamic; Gyro File hidden; the zoom Smoothness is greyed out with Zoom Mode Fixed')
    host.set(**STAB_TEST)

    # No .FPG next to the clip: off, and the image is the one the plug-in gave before.
    source(host, frames[0])
    host.set(decodeQuality=1)
    has_sidecar = any(CLIP.glob('*.FPG')) or any(CLIP.glob('*.fpg'))
    if not has_sidecar:
        for _ in range(100):                                # the look into the frames runs in the background
            host.changed('stabEnable')
            if not host.get('stabStatus').startswith('Loading'):
                break
        assert host.get('stabStatus') == 'Off: no gyro data in the DNGs, no .FPG gyro file next to the DNGs', host.get('stabStatus')
        st, a = host.render(source_frame=100)
        host.set(stabEnable=0)
        st2, b = host.render(source_frame=100)
        host.set(stabEnable=1)
        assert st == OK and st2 == OK and np.array_equal(a.pixels, b.pixels)
        ok('no .FPG in the clip folder: status says so, image identical to stabilisation switched off')
    host.set(stabEnable=0)
    st, plain = host.render(source_frame=100)
    assert st == OK
    host.set(stabEnable=1)
    P = green(plain)
    win = (slice(340, 740), slice(510, 1110))               # central 600 x 400 canvas pixels

    # Synthetic files (3264x2170 raster like the clip): frame 100 turned by 1.2 degrees.
    for label, exp in (('pan', (+d_px, 0.0)), ('tilt', (0.0, +d_px))):
        host.set(gyroFile=str(tmp / f'{label}.FPG'), stabAutoZoom=0)
        host.changed('gyroFile')
        assert host.get('stabStatus').startswith(f'On | {label}.FPG: 200 frames, 2500.00 Hz'), host.get('stabStatus')
        st, out = host.render(source_frame=100)
        assert st == OK
        dx, dy, peak = phase_shift(P[win], green(out)[win])
        assert peak > 0.2 and abs(dx * S - exp[0]) < 1.0 and abs(dy * S - exp[1]) < 1.0, (label, dx * S, dy * S, peak, exp)
        ok(f'GPU warp, {label} 1.2 deg: picture moves by the expected amount (raster px)',
           expected=[float(exp[0]), float(exp[1])], measured=[dx * S, dy * S])
    # A frame that was not displaced is nearly untouched (only the small smoothing share).
    st, out = host.render(source_frame=30)
    host.set(stabEnable=0)
    st2, ref = host.render(source_frame=30)
    host.set(stabEnable=1)
    assert st == OK and st2 == OK and float(np.abs(green(out) - green(ref)).max()) < 1e-4
    ok('GPU warp: a frame without rotation equals the unstabilised frame')

    # Rolling shutter on the GPU: steady 100 deg/s pan, smoothness 0 -> only the shear is removed.
    host.set(gyroFile=str(tmp / 'rs.FPG'), stabSmoothness=0.0)
    host.changed('gyroFile')
    st, out = host.render(source_frame=60)
    st2, plain60 = (None, None)
    host.set(stabEnable=0)
    st2, plain60 = host.render(source_frame=60)
    host.set(stabEnable=1)
    assert st == OK and st2 == OK
    A, B = green(plain60), green(out)
    meas, expd = [], []
    for rows in (slice(100, 164), slice(508, 572), slice(916, 980)):
        dx, dy, peak = phase_shift(A[rows, 410:1210], B[rows, 410:1210])
        y = (np.arange(rows.start, rows.stop) + 0.5) * S + ACTIVE[1]
        wgt = np.hanning(rows.stop - rows.start)
        expd.append(float((F_PX * np.tan(np.radians(100.0 * (y / RASTER[1] - 0.5) * READOUT)) * wgt).sum() / wgt.sum()))
        meas.append(dx * S)
        assert peak > 0.1 and abs(dy * S) < 1.0, (dx, dy, peak)
    assert max(abs(m - x) for m, x in zip(meas, expd)) < 1.5 and meas[0] < -30 and meas[2] > 30, (meas, expd)
    host.set(stabRollingShutter=0.0)
    st, flat = host.render(source_frame=60)
    assert st == OK and float(np.abs(green(flat) - A).max()) < 1e-4       # nothing left to correct
    host.set(stabRollingShutter=1.0, stabSmoothness=0.5)
    ok('GPU rolling shutter: top, middle and bottom bands are sheared back by the model amount (raster px)',
       expected_top_middle_bottom=expd, measured_top_middle_bottom=meas)

    # Wrong raster -> off, picture untouched.
    g = np.zeros((600 + 100 * 120, 3), np.int16)
    g[:, 1] = 13100
    host.set(gyroFile=str(write_fpg(tmp / 'small.FPG', g, 120, raster=(3024, 2010), active=(8, 5, 3008, 2000))))
    host.changed('gyroFile')
    assert host.get('stabStatus') == 'Off: small.FPG is for a 3024x2010 raster, the DNG is 3264x2170', host.get('stabStatus')
    st, out = host.render(source_frame=60)
    assert st == OK and np.array_equal(out.pixels, plain60.pixels)
    host.set(gyroFile=str(tmp / 'bad_truncated.FPG'))
    host.changed('gyroFile')
    assert host.get('stabStatus') == 'Off: bad_truncated.FPG: FPGY file is truncated', host.get('stabStatus')
    st, out = host.render(source_frame=60)
    assert st == OK and np.array_equal(out.pixels, plain60.pixels)
    ok('gyro file for another raster, or damaged: off with the reason, picture untouched')

    # Gyro data that ends before the clip (200 gyro frames, the clip has more DNGs).
    host.set(gyroFile=str(tmp / 'fade.FPG'))
    host.changed('gyroFile')
    assert f'gyro data ends at frame 200 of {len(frames)}: stabilisation fades out' in host.get('stabStatus'), host.get('stabStatus')
    pics = {}
    for fr in (150, 199, 200, 230):
        st, a = host.render(source_frame=fr)
        host.set(stabEnable=0)
        st2, b = host.render(source_frame=fr)
        host.set(stabEnable=1)
        assert st == OK and st2 == OK
        pics[fr] = float(np.abs(a.pixels - b.pixels).max())
    assert pics[150] > 0.05 and pics[199] < 0.02 and pics[200] == 0.0 and pics[230] == 0.0, pics
    ok('GPU: last gyro frame is the recorded picture, frames after it are untouched', max_pixel_difference_by_frame=pics)
    host.close()

    range_in_host(frames, tmp, ok)
    prefill_in_host(frames, tmp, ok)

    # Automatic pick-up: the sidecar next to the DNGs, nothing set by hand.
    clip = tmp / 'A001_092_auto'
    clip.mkdir()
    shutil.copy(frames[0], clip / frames[0].name)
    shutil.copy(SAMPLE, clip / 'A001_092.FPG')
    auto = OFXHost(PLUGIN, canvas=(1620, 1080))
    auto.set(**STAB_TEST)
    source(auto, clip / frames[0].name)
    status = auto.get('stabStatus')
    assert status.startswith('On | A001_092.FPG: 292 frames, 2499.46 Hz, readout 24.8 ms, 28.0 mm = 2520 px, zoom '), status
    st, on = auto.render(source_frame=0)
    auto.set(stabEnable=0)
    auto.changed('stabEnable')
    assert auto.get('stabStatus').startswith('Off (switched off) | A001_092.FPG')
    st2, off = auto.render(source_frame=0)
    assert st == OK and st2 == OK and not np.array_equal(on.pixels, off.pixels)
    auto.close()
    ok('sidecar next to the DNGs is found and used with default settings', status=status)


def prefill_in_host(frames, tmp, ok):
    """Focal Length and Rolling Shutter ms show the clip's own values until the user enters one."""
    g, _ = pulse_clip(1, 1)
    other = write_fpg(tmp / 'lens50.FPG', g, 200, focal_um=50000, readout_us=20000)
    host = OFXHost(PLUGIN, canvas=(1620, 1080))
    host.set(**STAB_TEST)
    assert (host.get('stabFocal'), host.get('stabReadout'), host.get('stabFocalAuto'), host.get('stabReadoutAuto')) == (0.0, 0.0, 1, 1)
    props = {n: host.obj(host.obj(host.params()[n])['props'])['values'] for n in ('stabFocalAuto', 'stabReadoutAuto', 'stabReadout')}
    assert props['stabFocalAuto'].get('OfxParamPropSecret') == [1] and props['stabReadoutAuto'].get('OfxParamPropSecret') == [1]
    assert props['stabReadout'].get('OfxParamPropParent') == ['stab'] and not props['stabReadout'].get('OfxParamPropSecret', [0])[0]
    source(host, frames[0])
    host.set(decodeQuality=1, gyroFile=str(tmp / 'pan.FPG'), stabAutoZoom=0)
    host.changed('gyroFile')
    assert (host.get('stabFocal'), host.get('stabReadout')) == (28.0, 24.8) and 'override' not in host.get('stabStatus'), host.get('stabStatus')
    st, a = host.render(source_frame=100)
    # The user types a focal length: it is his from now on.
    host.set(stabFocal=35.0)
    host.changed('stabFocal')
    assert host.get('stabFocalAuto') == 0 and '35.0 mm (override) = 3500 px' in host.get('stabStatus'), host.get('stabStatus')
    st2, b = host.render(source_frame=100)
    assert st == OK and st2 == OK and not np.array_equal(a.pixels, b.pixels)
    host.changed('stabEnable')
    assert host.get('stabFocal') == 35.0
    # Another clip: the field that is the clip's follows, the one the user set does not.
    host.set(gyroFile=str(other))
    host.changed('gyroFile')
    assert (host.get('stabFocal'), host.get('stabReadout')) == (35.0, 20.0), (host.get('stabFocal'), host.get('stabReadout'))
    # 0 (or the control's reset) gives the field back to the clip.
    host.set(stabFocal=0.0)
    host.changed('stabFocal')
    assert host.get('stabFocalAuto') == 1 and host.get('stabFocal') == 50.0 and 'override' not in host.get('stabStatus')
    # Rolling Shutter ms the same way.
    host.set(stabReadout=12.4)
    host.changed('stabReadout')
    assert host.get('stabReadoutAuto') == 0 and 'readout 12.4 ms (override)' in host.get('stabStatus'), host.get('stabStatus')
    host.set(gyroFile=str(tmp / 'pan.FPG'))
    host.changed('gyroFile')
    assert (host.get('stabFocal'), host.get('stabReadout')) == (28.0, 12.4)
    host.set(stabReadout=0.0)
    host.changed('stabReadout')
    assert host.get('stabReadoutAuto') == 1 and host.get('stabReadout') == 24.8
    st, c = host.render(source_frame=100)
    assert st == OK and np.array_equal(c.pixels, a.pixels)                # back to the clip's values: the same picture
    host.close()
    # A reopened project: the stored override is kept, a stored clip value is brought up to date.
    host = OFXHost(PLUGIN, canvas=(1620, 1080))
    host.set(**STAB_TEST)
    host.set(stabFocal=35.0, stabFocalAuto=0, stabReadout=99.0, stabReadoutAuto=1, gyroFile=str(tmp / 'pan.FPG'))
    source(host, frames[0])
    assert (host.get('stabFocal'), host.get('stabFocalAuto'), host.get('stabReadout'), host.get('stabReadoutAuto')) == (35.0, 0, 24.8, 1)
    assert '35.0 mm (override)' in host.get('stabStatus') and 'readout 24.8 ms,' in host.get('stabStatus')
    host.close()
    ok('Focal Length and Rolling Shutter ms: prefilled from the clip, a typed value is kept (also after reopening), 0 returns to the clip',
       prefilled=[28.0, 24.8], other_clip=[50.0, 20.0])


def mq_clips(ok, skip):
    """A001_001 / A001_002 (MQ 3024x2010, 50 fps, 1/640 s, in-frame gyro; A001_002 has late frame marks):
    picture shift between frames against the gyro, with each frame's own mark and with the regular cadence."""
    root = Path(os.environ.get('SFP_MQ_DIR', 'E:/CINEMA'))
    clips = {n: sorted((root / n).glob('*.DNG')) for n in ('A001_001', 'A001_002')}
    if any(len(v) < 1300 for v in clips.values()):
        return skip('MQ 50 clips: late marks, sync and image scale against the picture', f'{root} is not connected')

    def measure(frames, first, count):
        host = OFXHost(PLUGIN, canvas=(1504, 1000))
        host.set(**STAB_TEST)
        source(host, frames[0])
        host.set(decodeQuality=1, stabEnable=0)
        prev, out = None, []
        for i in range(first, first + count):
            st, img = host.render(source_frame=i)
            assert st == OK, host.messages[-2:]
            g = green(img)[250:750, 376:1128]
            if prev is not None:
                out.append(phase_shift(prev, g))
            prev = g
        host.close()
        return np.array(out)

    def motion(q, f):
        w1, x1, y1, z1 = (q[1:] * np.array([1, -1, -1, -1])).T
        w2, x2, y2, z2 = q[:-1].T
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        vz = 1 - 2 * (x * x + y * y)
        return f * np.stack([2 * (x * z + w * y) / vz, 2 * (y * z - w * x) / vz], 1)

    # The plug-in on a real in-frame MQ clip, everything at its default.
    host = OFXHost(PLUGIN, canvas=(1504, 1000))
    host.set(**STAB_TEST)
    source(host, clips['A001_001'][0])
    for _ in range(200):
        host.changed('stabEnable')
        if not host.get('stabStatus').startswith('Loading'):
            break
    status = host.get('stabStatus')
    assert status.startswith('On | gyro: 1740 frames, in-frame, 2499.49 Hz, readout 12.4 ms, 28.0 mm = 2333 px, zoom ') and \
        '| sync 7.0 ms = readout/2 + exposure/2 (1/640 s) | range: whole clip' in status, status
    assert (host.get('stabFocal'), host.get('stabReadout')) == (28.0, 12.42)
    host.set(decodeQuality=1)
    edges = 0
    for i in range(200, 1700, 150):
        st, out = host.render(source_frame=i)
        host.set(stabEnable=0)
        st2, plain = host.render(source_frame=i)
        host.set(stabEnable=1)

        def black(img):                                     # black edge pixels (this canvas has a 2 px pillarbox of its own)
            rgb = img.pixels[..., :3]
            return int(np.all(np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]) == 0, axis=1).sum())
        edges += int(black(out) > black(plain) + 20)
        assert st == OK and st2 == OK and not np.array_equal(out.pixels, plain.pixels)
    host.close()
    assert edges == 0
    ok('A001_001 in the plug-in with defaults: in-frame gyro found, fields prefilled, 10 frames stabilised without a border', status=status)

    # A001_002: around its late marks.
    first, count = 300, 900
    sh = measure(clips['A001_002'], first, count)
    good = sh[:, 2] > 0.05
    meas = sh[:, :2] * 2
    res = {}
    for label, o in (('own_marks', ['marks=raw']), ('regular_cadence', [])):
        j = gyro(clips['A001_002'][0], 'path=1', 'autozoom=0', *o)
        P = np.array(j['path'])
        res[label] = np.hypot(*(meas - motion(P[first:first + count, 1:5], j['focal_px'])).T)
    late = (P[:, 11] - P[:, 12]) > 0.005
    idx = np.arange(first, first + count - 1)
    touch = (late[idx] | late[idx + 1]) & good
    stats = {k: dict(median=float(np.median(v[touch])), rms=float(np.sqrt((v[touch] ** 2).mean())), max=float(v[touch].max())) for k, v in res.items()}
    rest = {k: float(np.median(v[good & ~touch])) for k, v in res.items()}
    assert j['late_marks'] == 11 and j['dropped_frames'] == 0 and '| sync 7.0 ms = readout/2 + exposure/2 (1/640 s) |' in j['status'], j['status']
    assert touch.sum() >= 15 and stats['regular_cadence']['median'] < 0.5 * stats['own_marks']['median'] and stats['regular_cadence']['rms'] < 0.6 * stats['own_marks']['rms'], stats
    ok('A001_002: gyro-predicted against measured picture shift on the frame pairs at its late marks (raster px)',
       late_marks=j['late_marks'], latest_ms=float(1e3 * (P[:, 11] - P[:, 12]).max()), pairs=int(touch.sum()), residual_at_late_marks=stats,
       residual_median_elsewhere=rest, rms_picture_shift_px=float(np.sqrt((meas[good] ** 2).sum(1).mean())))

    # A001_001: sync and image scale in a mode other than HQ 25.
    first, count = 100, 900
    sh = measure(clips['A001_001'], first, count)
    good = sh[:, 2] > 0.05
    meas = sh[:, :2] * 2
    curve = {}
    for ms in range(-8, 9, 2):
        j = gyro(clips['A001_001'][0], 'path=1', 'autozoom=0', f'sync={ms}')
        pred = motion(np.array(j['path'])[first:first + count, 1:5], j['focal_px'])
        e = np.hypot(*(meas - pred).T)[good]
        curve[ms] = (float(np.median(e)), float(np.sqrt((e ** 2).mean())))
        if ms == 0:
            slope = [float(np.polyfit(pred[good, k], meas[good, k], 1)[0]) for k in (0, 1)]
            delay, focal = j['delay_ms'], j['focal_px']
    fit = np.polyfit(list(curve), [v[1] ** 2 for v in curve.values()], 2)
    best = float(-fit[1] / (2 * fit[0]))
    assert abs(best) < 1.5 and all(abs(k - 1) < 0.03 for k in slope) and abs(delay - (12.423 / 2 + 1000 / 640 / 2)) < 0.01, (best, slope, delay)
    ok('A001_001 (MQ 50, 1/640 s): residual against Sync Offset has its minimum at 0; image scale right',
       delay_in_use_ms=delay, block_mark_delay_ms=11.211, best_sync_offset_ms=best, picture_says_delay_ms=delay + best,
       residual_median_rms_by_offset_ms={f'{k:+d}': [round(a, 3), round(b, 3)] for k, (a, b) in curve.items()},
       picture_against_gyro_slope_xy=slope, focal_px=focal)


def range_in_host(frames, tmp, ok):
    """Range: Whole Clip / Automatic / Manual, and the dynamic zoom, through the plug-in."""
    rough = tmp / 'rough.FPG'
    ref_whole = gyro(rough, 'clip=400')
    ref_part = gyro(rough, 'range=1-140', 'clip=400')
    assert ref_whole['zoom'] == 1.3 and ref_part['zoom'] < 1.08

    def zoom_of(status):
        return float(status.split(', zoom ')[1].split(' ')[0])

    def black_edge(out):
        rgb = out.pixels[..., :3]
        edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
        return int(np.all(edge == 0, axis=1).sum() > 20)

    host = OFXHost(PLUGIN, canvas=(1620, 1080))
    host.set(**STAB_TEST)
    source(host, frames[0])
    host.set(decodeQuality=1, gyroFile=str(rough))
    host.changed('gyroFile')
    # Automatic, before any frame: nothing known yet, whole clip.
    s0 = host.get('stabStatus')
    assert s0.endswith('| range: whole clip (automatic: known once a frame has been rendered)') and zoom_of(s0) == 1.3, s0
    # An untrimmed clip: the host's clip range covers everything.
    st, _ = host.render(source_frame=20)
    host.changed('stabEnable')
    s1 = host.get('stabStatus')
    assert st == OK and '| range: whole clip (automatic: host clip range 0 to 999999, source frame = time +1)' in s1 and zoom_of(s1) == 1.3, s1
    # The timeline clip uses source frames 1..140 (effect time 0..139).
    host.set_clip_range(0, 139)
    borders = 0
    for i in (0, 60, 139):
        st, out = host.render(source_frame=i)
        assert st == OK
        borders += black_edge(out)
    host.changed('stabEnable')
    s2 = host.get('stabStatus')
    assert '| range: frames 1 to 140 of 400 (automatic: host clip range 0 to 139, source frame = time +1)' in s2, s2
    assert abs(zoom_of(s2) - ref_part['zoom']) < 1e-3 and borders == 0
    host.close()
    # A piece cut from the middle: effect time 0..99 shows source frames 231..330.
    host = OFXHost(PLUGIN, canvas=(1620, 1080))
    host.set(**STAB_TEST)
    source(host, frames[0])
    host.set(decodeQuality=1, gyroFile=str(rough))
    host.set_clip_range(0, 99)
    for t in (0, 50, 99):
        st, out = host.render(time=float(t), source_frame=230 + t)
        assert st == OK and black_edge(out) == 0
    host.changed('stabEnable')
    s3 = host.get('stabStatus')
    assert '| range: frames 231 to 330 of 400 (automatic: host clip range 0 to 99, source frame = time +231)' in s3, s3
    assert abs(zoom_of(s3) - gyro(rough, 'range=231-330', 'clip=400')['zoom']) < 1e-3
    # A host range that does not contain the frame being rendered is not used.
    host.set_clip_range(1000, 1100)
    host.render(time=50.0, source_frame=280)
    host.changed('stabEnable')
    assert host.get('stabStatus').endswith('| range: whole clip (automatic: the host reports no clip range)'), host.get('stabStatus')
    # Time and source frame no longer move together (a retime): whole clip.
    host.set_clip_range(0, 99)
    host.render(time=50.0, source_frame=300)
    host.changed('stabEnable')
    assert host.get('stabStatus').endswith('| range: whole clip (automatic: the clip is retimed, whole clip used)'), host.get('stabStatus')
    # Whole Clip ignores the host; Manual uses the two frame numbers.
    host.set(stabRange=0)
    host.changed('stabRange')
    assert host.get('stabStatus').endswith('| range: whole clip') and zoom_of(host.get('stabStatus')) == 1.3
    host.set(stabRange=2, stabRangeStart=1.0, stabRangeEnd=140.0)
    host.changed('stabRange')
    s4 = host.get('stabStatus')
    assert s4.endswith('| range: frames 1 to 140 of 400 (manual)') and abs(zoom_of(s4) - ref_part['zoom']) < 1e-3, s4
    st, a = host.render(source_frame=60)
    host.set(stabRange=0)
    st2, b = host.render(source_frame=60)
    assert st == OK and st2 == OK and not np.array_equal(a.pixels, b.pixels)             # less zoom than with the whole clip
    host.set(stabRange=2, stabRangeStart=230.0, stabRangeEnd=0.0)
    host.changed('stabRange')
    assert host.get('stabStatus').endswith('| range: frames 230 to 400 of 400 (manual)')
    host.set(stabRangeStart=300.0, stabRangeEnd=200.0)
    host.changed('stabRange')
    assert host.get('stabStatus').endswith('| range: whole clip (manual range is empty: whole clip used)')
    ok('range in the plug-in: Automatic follows the host clip range and the time-to-source mapping; Manual and Whole Clip; no border',
       automatic_untrimmed=s1.split(' | ')[-1], automatic_trimmed=s2.split(' | ')[-1], automatic_cut_from_the_middle=s3.split(' | ')[-1],
       zoom_whole=1.3, zoom_frames_1_to_140=zoom_of(s2), zoom_frames_231_to_330=zoom_of(s3))

    # Dynamic zoom on the GPU: no border on any frame through the rough part, picture scale changes slowly.
    host.set(stabRange=0, stabZoomMode=1, stabZoomSmooth=2.0, stabMaxZoom=3.0)
    host.changed('stabZoomMode')
    sd = host.get('stabStatus')
    assert '(dynamic, mean ' in sd, sd
    z = np.array(gyro(rough, 'path=1', 'clip=400', 'zoommode=dynamic', 'zoomsmooth=2', 'maxzoom=3')['path'])[:, 9]
    edges = 0
    for i in range(96, 260, 4):
        st, out = host.render(source_frame=i)
        assert st == OK
        edges += black_edge(out)
    host.set(stabZoomMode=0)
    st, fixed = host.render(source_frame=20)
    host.set(stabZoomMode=1)
    st2, dyn = host.render(source_frame=20)
    assert edges == 0 and st == OK and st2 == OK and not np.array_equal(fixed.pixels, dyn.pixels)
    host.close()
    ok('dynamic zoom in the plug-in: no border on 41 frames through the rough part', status=sd.split(', zoom ')[1].split(' | ')[0],
       zoom_at_frames_20_100_175_250=[float(z[i]) for i in (20, 100, 175, 250)], largest_change_per_second=float(np.abs(np.diff(z)).max() * 25))


def real_clip(frames, ok):
    """Frames 1..290 of A001_092 (25 fps, 28 mm, large deliberate pans and tilts)."""
    if not frames[0].name.upper().startswith('A001_092'):
        return
    S = 2.0
    N = 290
    win = (slice(270, 810), slice(405, 1215))               # central half of the picture

    borders = {}

    def render(name=None, **values):
        host = OFXHost(PLUGIN, canvas=(1620, 1080))
        host.set(**STAB_TEST)
        source(host, frames[0])
        host.set(decodeQuality=1, gyroFile=str(SAMPLE), **values)
        seq = np.zeros((N, 540, 810), np.float32)
        black = 0
        for i in range(N):
            st, out = host.render(source_frame=i)
            assert st == OK, host.messages[-2:]
            seq[i] = green(out)[win]
            rgb = out.pixels[..., :3]
            edge = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
            black += int(np.all(edge == 0, axis=1).sum() > 20)        # a frame with border pixels outside the recording
        if name:
            borders[name] = black
        host.close()
        return np.array([phase_shift(seq[i - 1], seq[i]) for i in range(1, N)])

    def stats(v):
        m = np.hypot(v[:, 0], v[:, 1]) * S
        return dict(rms=float(np.sqrt((m ** 2).mean())), median=float(np.median(m)), p90=float(np.percentile(m, 90)))

    j = gyro(SAMPLE, 'path=1', 'autozoom=0')
    path = np.array(j['path'])[:N]
    # (The sample's gyro data ends at frame 292, so the last 25 frames are inside its fade-out;
    # the virtual-camera path reported by the tool includes that.)

    def centre_motion(q):                                   # picture motion at the centre, raster px
        w1, x1, y1, z1 = (q[1:] * np.array([1, -1, -1, -1])).T
        w2, x2, y2, z2 = q[:-1].T
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        vz = 1 - 2 * (x * x + y * y)
        return j['focal_px'] * np.stack([2 * (x * z + w * y) / vz, 2 * (y * z - w * x) / vz], 1)

    def jitter(v, keep):                                    # change of the shift from one frame pair to the next
        m = np.hypot(*np.diff(v[:, :2] * S, axis=0).T)[keep[1:] & keep[:-1]]
        return dict(rms=float(np.sqrt((m ** 2).mean())), median=float(np.median(m)))

    raw = render('unstabilised', stabEnable=0)
    stab = render('zoom_1', stabAutoZoom=0)                 # default smoothness 0.5 s, rolling shutter 1, sync 0
    firm = render(stabAutoZoom=0, stabSmoothness=2.0)
    # Pairs with a clear correlation peak in all three (fast moves blur the frames: 1/100 s).
    good = (raw[:, 2] > 0.03) & (stab[:, 2] > 0.03) & (firm[:, 2] > 0.03)
    r, s, s2 = stats(raw[good]), stats(stab[good]), stats(firm[good])
    jr, js = jitter(raw, good), jitter(stab, good)
    # Stabilised motion that is not the smoothed camera path = what the stabilisation got wrong.
    err = stab[:, :2] * S - centre_motion(path[:, 5:9])
    e = stats(err[good] / S)
    pred = centre_motion(path[:, 1:5])
    fit = [float(np.polyfit(pred[good, k], raw[good, k] * S, 1)[0]) for k in (0, 1)]
    corr = [float(np.corrcoef(pred[good, k], raw[good, k])[0, 1]) for k in (0, 1)]
    assert corr[0] > 0.99 and corr[1] > 0.98, corr
    assert s['rms'] < 0.35 * r['rms'] and s2['rms'] < 0.35 * r['rms'] and s2['median'] < 0.5 * r['median'], (r, s, s2)
    assert js['median'] < 0.25 * jr['median'] and js['rms'] < 0.35 * jr['rms'], (jr, js)
    assert e['median'] < 2.5, e
    ok('A001_092 frames 1..290: frame-to-frame picture shift, unstabilised against stabilised at zoom 1 (raster px)',
       pairs=int(good.sum()), of=N - 1, unstabilised=r, stabilised_default_0_5s=s, stabilised_smoothness_2s=s2,
       jitter_unstabilised=jr, jitter_stabilised=js, stabilised_minus_smoothed_path=e,
       gyro_vs_image_slope_xy=fit, gyro_vs_image_correlation_xy=corr, focal_px=j['focal_px'])

    # Everything at its default (auto zoom, limit 1.3): what the user gets without touching anything.
    jd = gyro(SAMPLE, 'path=1')
    dflt = render('defaults', )
    gd = good & (dflt[:, 2] > 0.03)
    zd = np.array(jd['path'])[:N, 9]
    dflt[:, :2] /= (0.5 * (zd[1:] + zd[:-1]))[:, None]      # back to raster px
    sd, jdf = stats(dflt[gd]), jitter(dflt, gd)
    ed = stats((dflt[:, :2] * S - centre_motion(np.array(jd['path'])[:N, 5:9]))[gd] / S)
    assert borders == {'unstabilised': 0, 'zoom_1': borders['zoom_1'], 'defaults': 0} and borders['zoom_1'] > 50, borders
    assert jdf['median'] < 0.5 * jr['median'] and sd['rms'] < 0.6 * r['rms'] and ed['median'] < 2.5, (sd, jdf, ed)
    ok('A001_092 frames 1..290 with every default (auto zoom at its 1.3 limit on this violent take), raster px',
       pairs=int(gd.sum()), zoom=jd['zoom'], frames_with_reduced_stabilisation=jd['limited_frames'],
       frames_showing_a_border=borders, stabilised=sd, jitter_stabilised=jdf, stabilised_minus_virtual_camera_path=ed)

    # Rolling shutter: horizontal shift of the top band against the bottom band, per frame pair.
    def shear(**values):
        host = OFXHost(PLUGIN, canvas=(1620, 1080))
        host.set(**STAB_TEST)
        source(host, frames[0])
        host.set(decodeQuality=1, gyroFile=str(SAMPLE), **values)
        prev, out = None, []
        for i in range(N):
            st, img = host.render(source_frame=i)
            assert st == OK
            g = green(img)
            cur = (g[80:460, 405:1215], g[620:1000, 405:1215])
            if prev is not None:
                t, b = phase_shift(prev[0], cur[0]), phase_shift(prev[1], cur[1])
                out.append(((b[0] - t[0]) * S, min(t[2], b[2])))
            prev = cur
        host.close()
        return np.array(out)

    sh = {'unstabilised': shear(stabEnable=0), 'rolling_shutter_0': shear(stabAutoZoom=0, stabRollingShutter=0.0),
          'rolling_shutter_1': shear(stabAutoZoom=0)}
    keep = np.all([v[:, 1] > 0.05 for v in sh.values()], axis=0)
    med = {k: float(np.median(np.abs(v[keep, 0]))) for k, v in sh.items()}
    assert med['rolling_shutter_1'] < 0.5 * med['rolling_shutter_0'] and med['rolling_shutter_1'] < 0.5 * med['unstabilised'], med
    ok('A001_092: top-against-bottom horizontal shift per frame pair (median, raster px)', pairs=int(keep.sum()), **med)

    # Sync: the residual is smallest near the file's mark delay.
    res = {}
    for ms in (-6, -3, 0, 3, 6):
        v = render(stabAutoZoom=0, stabSync=float(ms))
        p = np.array(gyro(SAMPLE, 'path=1', 'autozoom=0', f'sync={ms}')['path'])[:N]
        k = good & (v[:, 2] > 0.03)
        res[ms] = float(np.median(np.hypot(*(v[k, :2] * S - centre_motion(p[:, 5:9])[k]).T)))
    assert min(res, key=res.get) in (0, 3) and res[0] < res[-6] and res[0] < res[6], res
    ok('A001_092: median residual against the smoothed path by sync offset (ms -> raster px)', **{f'sync_{k:+d}ms': v for k, v in res.items()})


if __name__ == '__main__':
    main()
