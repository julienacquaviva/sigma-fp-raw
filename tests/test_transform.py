"""Tests of Fit, Transform (1.5.0) and the resampling kernel (1.5.1): geometry against an independent
model of the controls, pictures against one-step references built from the full-resolution developed
frame, and the quality of the kernel (Lanczos-3, widened for downscales) against bilinear.

Runs without camera files: the frames are synthetic DNGs (tests/make_test_dng.py).
usage: python tests/test_transform.py            (GPU, then the processor path in a second process)
       SFP_REFERENCE_OFX=<1.4.2 SigmaFpRaw.ofx> python tests/test_transform.py   adds: neutral = 1.4.2, bit for bit"""
import hashlib, json, math, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from ofx_host import OFXHost, OK  # noqa: E402

# The values these tests were written for (the defaults until 1.5.1; since 1.6.0: off, 0.1 s, Dynamic).
STAB_TEST = dict(stabEnable=1, stabSmoothness=0.5, stabZoomMode=0, developRaw=1)   # and the plug-in's own development (off by default since 1.9.2)
import make_test_dng as MD  # noqa: E402

PLUGIN = ROOT / 'dist/SigmaFpRaw.ofx.bundle/Contents/Win64/SigmaFpRaw.ofx'
CLI = ROOT / 'build/sfp_cli.exe'
SCRATCH = HERE / 'scratch/transform'
CPU_PART = '--cpu-part' in sys.argv
os.environ['SFP_GYRO_CACHE_DIR'] = 'off'
NEUTRAL = dict(zx=1.0, zy=1.0, px=0.0, py=0.0, rot=0.0, ax=0.0, ay=0.0, pitch=0.0, yaw=0.0, fliph=False, flipv=False)


# ---- the independent model ----

def fit_of(cw, ch, W, H, fit):
    sx, sy = W / cw, H / ch
    s = {'fit': min(sx, sy), 'width': sx, 'height': sy}[fit]
    return s, (W - cw * s) / 2, (H - ch * s) / 2


def inverse(ix, iy, W, H, xf, rs=1.0):
    """Output position (y down, pixel centres at +0.5) -> fitted frame position, undoing the steps
    one at a time in reverse: position and anchor, perspective, rotation, zoom, anchor, flip."""
    x = dict(NEUTRAL, **xf)
    u = np.asarray(ix, float) - W / 2
    v = H / 2 - np.asarray(iy, float)
    ax, ay = x['ax'] * rs, x['ay'] * rs
    u = u - (ax + x['px'] * rs)
    v = v - (ay + x['py'] * rs)
    ok = np.ones(np.shape(u), bool)
    if x['pitch'] or x['yaw']:
        p, y, D = math.radians(x['pitch']), math.radians(x['yaw']), W
        X0, Y0, Z0 = math.cos(y), 0.0, math.sin(y)
        X1, Y1, Z1 = -math.sin(y) * math.sin(p), math.cos(p), math.cos(y) * math.sin(p)
        a, b = D * X0 - u * Z0, D * X1 - u * Z1
        c, d = D * Y0 - v * Z0, D * Y1 - v * Z1
        det = a * d - b * c
        px_ = (u * D * d - b * v * D) / det
        py_ = (a * v * D - c * u * D) / det
        ok = (D + Z0 * px_ + Z1 * py_) > 0
        u, v = px_, py_
    r = math.radians(x['rot'])
    c, s = math.cos(r), math.sin(r)
    u, v = c * u + s * v, -s * u + c * v
    u, v = u / x['zx'], v / x['zy']
    u, v = u + ax, v + ay
    if x['fliph']:
        u = -u
    if x['flipv']:
        v = -v
    return u + W / 2, H / 2 - v, ok


def to_crop(ix, iy, cw, ch, W, H, fit, xf, rs=1.0):
    fx, fy, ok = inverse(ix, iy, W, H, xf, rs)
    s, ox, oy = fit_of(cw, ch, W, H, fit)
    return (fx - ox) / s, (fy - oy) / s, ok


def cli_map(cw, ch, W, H, fit, xf, pts, scale=1.0):
    x = dict(NEUTRAL, **xf)
    args = [str(CLI), 'map', str(cw), str(ch), str(W), str(H), f'fit={fit}', f'scale={scale}'] + (
        [f'{k}={int(v) if isinstance(v, bool) else v}' for k, v in x.items()] + [f'pt={a},{b}' for a, b in pts])
    return json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)


def bilinear(img, sx, sy):
    """img: developed full-resolution crop, pixel k centred at k + 0.5 (the kernels' convention)."""
    h, w = img.shape[:2]
    fx, fy = sx - 0.5, sy - 0.5
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = (fx - x0)[..., None], (fy - y0)[..., None]
    x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
    x0, y0 = np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1)
    return (img[y0, x0] * (1 - tx) * (1 - ty) + img[y0, x1] * tx * (1 - ty) + img[y1, x0] * (1 - tx) * ty + img[y1, x1] * tx * ty)


def lanczos(t, a=3.0):
    t = np.abs(t)
    out = np.where(t < 1e-6, 1.0, a * np.sin(np.pi * t) * np.sin(np.pi * t / a) / np.maximum((np.pi * t) ** 2, 1e-30))
    return np.where(t >= a, 0.0, out)


def camera_matrix(dng):
    """Camera -> output matrix of the frame (the anti-ringing limit acts on the camera planes, before it)."""
    out = subprocess.run([str(CLI), 'develop', str(dng), str(SCRATCH / 'm.tif'), 'width=64', 'height=64', 'stab=0'], capture_output=True, text=True).stdout
    line = [ln for ln in out.splitlines() if ln.startswith('matrix:')][0]
    return np.array([float(v) for v in line.split()[1:]]).reshape(3, 3)


def lanczos_camera(full, M, sx, sy, kx, ky):
    """lanczos_sample in the camera planes: back through the matrix, sample, forward again."""
    cam = full @ np.linalg.inv(M).T
    return lanczos_sample(cam, sx, sy, kx, ky) @ M.T


def lanczos_sample(img, sx, sy, kx, ky, a=3.0):
    """The reference for the Best kernel, written independently: separable Lanczos-3 about (sx, sy),
    widened by kx / ky (source pixels per output pixel, at least 1), coordinates clamped to the frame,
    weights normalised; at 1:1 or enlarging, held to the range of the 2x2 nearest pixels."""
    h, w = img.shape[:2]
    fx, fy = np.asarray(sx) - 0.5, np.asarray(sy) - 0.5
    n = len(fx)
    rx, ry = int(np.ceil(a * kx)) + 1, int(np.ceil(a * ky)) + 1
    xs = np.floor(fx).astype(int)[:, None] + np.arange(-rx, rx + 1)[None, :]
    ys = np.floor(fy).astype(int)[:, None] + np.arange(-ry, ry + 1)[None, :]
    wx = lanczos((xs - fx[:, None]) / kx, a)
    wy = lanczos((ys - fy[:, None]) / ky, a)
    acc = np.zeros((n, img.shape[2]))
    for j in range(ys.shape[1]):
        rows = img[np.clip(ys[:, j], 0, h - 1)[:, None], np.clip(xs, 0, w - 1)]          # n, taps, 3
        acc += wy[:, j, None] * (rows * wx[:, :, None]).sum(1)
    out = acc / (wx.sum(1) * wy.sum(1))[:, None]
    if kx <= 1 and ky <= 1:
        x0, y0 = np.clip(np.floor(fx).astype(int), 0, w - 1), np.clip(np.floor(fy).astype(int), 0, h - 1)
        x1, y1 = np.clip(np.floor(fx).astype(int) + 1, 0, w - 1), np.clip(np.floor(fy).astype(int) + 1, 0, h - 1)
        four = np.stack([img[y0, x0], img[y0, x1], img[y1, x0], img[y1, x1]])
        out = np.clip(out, four.min(0), four.max(0))
    return out


# ---- plug-in helpers ----

XF_PARAMS = dict(zx='xfZoomX', zy='xfZoomY', px='xfPosX', py='xfPosY', rot='xfRotation', ax='xfAnchorX', ay='xfAnchorY',
                 pitch='xfPitch', yaw='xfYaw', fliph='xfFlipH', flipv='xfFlipV')


def render(dng, canvas, xf=None, fit=0, plugin=PLUGIN, extra=None, frame=0, fast=False):
    host = OFXHost(plugin, canvas=canvas)
    host.set(**STAB_TEST)
    host.obj(host.obj(host.instance)['props'])['values']['OfxImageEffectPropSrcFilePath'] = [str(dng)]
    host.changed('Source')
    values = dict(gamma=0, sharpness=0.0, stabEnable=0)
    if plugin == PLUGIN:
        values.update(fitMode=fit, xfZoomLink=0, resampling=1 if fast else 0)
        for k, v in (xf or {}).items():
            values[XF_PARAMS[k]] = int(v) if isinstance(v, bool) else float(v)
    values.update(extra or {})
    host.set(**values)
    st, out = host.render(source_frame=frame)
    assert st == OK, host.messages[-2:]
    pix = out.pixels[::-1, :, :3].astype(np.float64).copy()      # top-down
    host.close()
    return pix


def main():
    report = []

    def ok(name, **info):
        report.append(dict(name=name, passed=True, **info))
        print('PASS', name, info if info else '')

    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    cw, ch = 2160, 1440                                    # the R124 S16 recording size
    dng = MD.write_dng(SCRATCH / 'SYN_S16_000001.DNG', cw, ch)

    if not CPU_PART:
        geometry(ok)
    pixels(ok, dng, cw, ch)
    if not CPU_PART:
        stabilised(ok, dng, cw, ch)
        quality(ok)
        params(ok, dng, cw, ch)
        reference_build(ok)
        r = subprocess.run([sys.executable, __file__, '--cpu-part'], env=dict(os.environ, SFP_CPU='1'), capture_output=True, text=True)
        print(r.stdout[-3000:])
        assert r.returncode == 0 and 'ALL PASS' in r.stdout, r.stderr[-2000:]
        ok('the same picture tests on the processor path (SFP_CPU=1)', result=r.stdout.strip().splitlines()[-1])
    shutil.rmtree(SCRATCH, ignore_errors=True)
    if not CPU_PART:
        (ROOT / 'tests/transform_report.json').write_text(json.dumps({'plugin': str(PLUGIN), 'tests': report}, indent=2, default=str) + '\n')
    print(f'ALL PASS ({len(report)})')


def geometry(ok):
    # Fit Width / Fit Height / Scale to Fit for the R124 sizes into UHD and DCI 4K timelines.
    rows = {}
    for cw, ch in ((3840, 2560), (4320, 2160), (3600, 1800), (2160, 1440)):
        for W, H in ((3840, 2160), (4096, 2160)):
            for fit in ('fit', 'width', 'height'):
                s, ox, oy = fit_of(cw, ch, W, H, fit)
                pts = [(0.5, 0.5), (W / 2, H / 2), (W - 0.5, H - 0.5)]
                r = cli_map(cw, ch, W, H, fit, {}, pts)
                assert abs(r['scale'][0] - s) < 1e-6 and abs(r['scale'][1] - s) < 1e-6 and abs(r['offset'][0] - ox) < 1e-3 and abs(r['offset'][1] - oy) < 1e-3
                c = r['points'][1]
                assert abs(c[2] - cw / 2) < 1e-3 and abs(c[3] - ch / 2) < 1e-3                       # centre to centre
                if fit == 'width':
                    assert abs(r['points'][0][2] - 0.5 / s) < 1e-3 and abs(r['points'][2][2] - (cw - 0.5 / s)) < 1e-2    # edge to edge
                if fit == 'height':
                    assert abs(r['points'][0][3] - 0.5 / s) < 1e-3 and abs(r['points'][2][3] - (ch - 0.5 / s)) < 1e-2
                rows[f'{cw}x{ch} -> {W}x{H} {fit}'] = round(s, 6)
    ok('Fit Width / Fit Height / Scale to Fit: scale, centring and edges for the R124 sizes into 3840x2160 and 4096x2160', **rows)

    # Each control against the independent model, at random output points, and what it means on screen.
    rng = np.random.default_rng(3)
    W, H, cw, ch = 3840, 2160, 3840, 2560
    pts = [(float(a), float(b)) for a, b in zip(rng.uniform(0, W, 40), rng.uniform(0, H, 40))]
    cases = {'zoom 1.7': dict(zx=1.7, zy=1.7), 'zoom x 1.3 y 0.8': dict(zx=1.3, zy=0.8), 'position 123.5, -45': dict(px=123.5, py=-45.0),
             'rotation 33': dict(rot=33.0), 'anchor -400, 250 with zoom 2 and rotation 15': dict(ax=-400.0, ay=250.0, zx=2.0, zy=2.0, rot=15.0),
             'pitch 20': dict(pitch=20.0), 'yaw -25': dict(yaw=-25.0), 'pitch 10 yaw 15 rotation 5 zoom 1.2': dict(pitch=10.0, yaw=15.0, rot=5.0, zx=1.2, zy=1.2),
             'flip horizontal': dict(fliph=True), 'flip vertical': dict(flipv=True), 'everything': dict(zx=1.4, zy=1.1, px=-60.0, py=33.0, rot=-12.0, ax=100.0, ay=-50.0, pitch=-8.0, yaw=6.0, fliph=True, flipv=True)}
    worst = {}
    for name, xf in cases.items():
        for fit in ('fit', 'width'):
            r = cli_map(cw, ch, W, H, fit, xf, pts)
            sx, sy, good = to_crop(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]), cw, ch, W, H, fit, xf)
            got = np.array([[q[2], q[3]] for q in r['points']])
            err = float(np.abs(got - np.stack([sx, sy], 1))[good].max())
            assert err < 0.02 and r['xform'] == 1, (name, fit, err)
            worst[name] = max(worst.get(name, 0), err)
    # The meaning of each control on screen (scale to fit of 3840x2560 into 3840x2160: s = 0.84375).
    s = 2160 / 2560

    def where(xf, ix, iy, scale=1.0, Wt=W, Ht=H):
        q = cli_map(cw, ch, Wt, Ht, 'fit', xf, [(ix, iy)], scale)['points'][0]
        return (q[2] - cw / 2) * s * scale, (q[3] - ch / 2) * s * scale    # fitted offset from the centre, output pixels, y down
    assert np.allclose(where(dict(px=100.0, py=50.0), W / 2 + 100, H / 2 - 50), (0, 0), atol=1e-3)             # right and up
    assert np.allclose(where(dict(rot=90.0), W / 2, H / 2 - 100), (100, 0), atol=1e-3)                         # anticlockwise
    assert np.allclose(where(dict(zx=2.0, zy=2.0, ax=200.0), W / 2 + 200, H / 2), (200, 0), atol=1e-3)        # the anchor stays put
    assert np.allclose(where(dict(zx=2.0, zy=2.0), W / 2 + 200, H / 2), (100, 0), atol=1e-3)
    assert np.allclose(where(dict(ax=300.0, ay=-90.0), W / 2 + 37, H / 2 + 11), (37, 11), atol=1e-3)          # the anchor alone moves nothing
    assert np.allclose(where(dict(fliph=True), W / 2 + 100, H / 2 + 20), (-100, 20), atol=1e-3)
    assert np.allclose(where(dict(flipv=True), W / 2 + 100, H / 2 + 20), (100, -20), atol=1e-3)
    top, bottom = where(dict(pitch=20.0), W / 2 + 500, H / 2 - 500), where(dict(pitch=20.0), W / 2 + 500, H / 2 + 500)
    assert abs(top[0]) > abs(bottom[0])                            # the top leans away: smaller on screen, so a point there reads further out
    right, left = where(dict(yaw=20.0), W / 2 + 800, H / 2 - 300), where(dict(yaw=20.0), W / 2 - 800, H / 2 - 300)
    assert abs(right[1]) > abs(left[1])                            # the right side turns away
    half = where(dict(px=100.0), W / 4 + 50, H / 4, scale=0.5, Wt=W / 2, Ht=H / 2)
    assert np.allclose(half, (0, 0), atol=1e-3)                    # proxy render at half size: 100 timeline px = 50 output px
    ok('each Transform control against an independent step-by-step model (largest difference, crop px), and its direction on screen',
       **{k: round(v, 5) for k, v in worst.items()})


def pixels(ok, dng, cw, ch):
    # The developed frame at full resolution: Scale to Fit into a canvas of the crop size = 1:1, no resampling.
    full = render(dng, (cw, ch))
    W, H = cw // 2, ch // 2                                        # a half-size timeline: Scale to Fit = 0.5
    path = 'processor' if CPU_PART else 'GPU'
    exact = {}
    # Transforms that land on the source pixel centres: the result must be the source pixels themselves.
    for name, xf in (('zoom 2', dict(zx=2.0, zy=2.0)), ('zoom 2, rotation 90', dict(zx=2.0, zy=2.0, rot=90.0)),
                     ('zoom 2, rotation 180, flip vertical', dict(zx=2.0, zy=2.0, rot=180.0, flipv=True)),
                     ('zoom 2, flip horizontal', dict(zx=2.0, zy=2.0, fliph=True)), ('zoom 2, position 7, -3', dict(zx=2.0, zy=2.0, px=7.0, py=-3.0)),
                     ('zoom 2, anchor 100, 40, rotation -90', dict(zx=2.0, zy=2.0, ax=100.0, ay=40.0, rot=-90.0))):
        out = render(dng, (W, H), xf)
        iy, ix = np.mgrid[0:H, 0:W] + 0.5
        sx, sy, good = to_crop(ix, iy, cw, ch, W, H, 'fit', xf)
        inside = good & (sx >= 0) & (sy >= 0) & (sx < cw) & (sy < ch)
        assert np.allclose(sx[inside] % 1, 0.5, atol=1e-4) and np.allclose(sy[inside] % 1, 0.5, atol=1e-4)
        ref = np.zeros_like(out)
        ref[inside] = full[np.floor(sy[inside]).astype(int), np.floor(sx[inside]).astype(int)]
        diff = float(np.abs(out - ref).max())
        assert diff < 1e-5 and inside.mean() > 0.2, (name, diff)
        assert np.all(out[~inside] == 0)
        exact[name] = diff
    ok(f'{path}: transforms that land on source pixels give the source pixels exactly (one resampling, none before it)', **exact)
    # Any other transform: one bilinear step from the full-resolution frame (linear output, so it commutes
    # with the colour processing). A two-step chain (fit first, then transform) is far off.
    one = {}
    W, H = 1600, 900
    for name, xf in (('rotation 30, zoom 1.3', dict(rot=30.0, zx=1.3, zy=1.3)), ('pitch 15, yaw -10', dict(pitch=15.0, yaw=-10.0, zx=1.2, zy=1.2)),
                     ('everything', dict(zx=1.6, zy=1.4, px=-20.25, py=9.5, rot=-7.0, ax=60.0, ay=-30.0, pitch=4.0, yaw=-3.0, fliph=True))):
        out = render(dng, (W, H), xf, fast=True)
        iy, ix = np.mgrid[0:H, 0:W] + 0.5
        sx, sy, good = to_crop(ix, iy, cw, ch, W, H, 'fit', xf)
        inside = good & (sx >= 1) & (sy >= 1) & (sx < cw - 1) & (sy < ch - 1)
        ref = bilinear(full, sx, sy)
        err = float(np.abs(out - ref)[inside].max())
        # two steps: the fitted frame first (bilinear at the fit), then the transform (bilinear again)
        s, ox, oy = fit_of(cw, ch, W, H, 'fit')
        fy0, fx0 = np.mgrid[0:H, 0:W] + 0.5
        fitted = bilinear(full, (fx0 - ox) / s, (fy0 - oy) / s)
        fx, fy, _ = inverse(ix, iy, W, H, xf)
        two = bilinear(fitted, fx, fy)
        err2 = float(np.abs(out - two)[inside & (fx > 1) & (fy > 1) & (fx < W - 1) & (fy < H - 1)].mean())
        assert err < 2e-4, (name, err)
        one[name] = dict(max_diff_to_one_step=err, mean_diff_to_two_steps=err2)
        assert err2 > 20 * float(np.abs(out - ref)[inside].mean()), (name, err2)
    ok(f'{path}: Fast (bilinear): other transforms equal one bilinear step from the full-resolution frame, not a two-step chain', **one)
    # Best: the same single step with the Lanczos-3 kernel, against an independent implementation, at
    # 3000 random output pixels; downscale (kernel widened), about 1:1 with rotation, and enlargement.
    best = {}
    M = camera_matrix(dng)
    rng = np.random.default_rng(11)
    for name, (W, H), xf in (('fit 2160x1440 into 1280x720 (x 0.5), rotation 12', (1280, 720), dict(rot=12.0)),
                             ('fit into 1280x720, zoom 0.7 x 0.9', (1280, 720), dict(zx=0.7, zy=0.9)),
                             ('fit 1:1, rotation 30', (cw, ch), dict(rot=30.0)),
                             ('fit 1:1, zoom 2.5, pitch 6', (cw, ch), dict(zx=2.5, zy=2.5, pitch=6.0, px=40.5))):
        out = render(dng, (W, H), xf)
        ix, iy = rng.integers(0, W, 3000) + 0.5, rng.integers(0, H, 3000) + 0.5
        sx, sy, good = to_crop(ix, iy, cw, ch, W, H, 'fit', xf)
        c = [to_crop(np.array([W / 2 + dx]), np.array([H / 2 + dy]), cw, ch, W, H, 'fit', xf) for dx, dy in ((0, 0), (1, 0), (0, 1))]
        kx = float(np.hypot(c[1][0] - c[0][0], c[2][0] - c[0][0])[0])
        ky = float(np.hypot(c[1][1] - c[0][1], c[2][1] - c[0][1])[0])
        kx, ky = (kx if kx > 1.001 else 1.0), (ky if ky > 1.001 else 1.0)
        inside = good & (sx >= 0) & (sy >= 0) & (sx < cw) & (sy < ch)
        ref = lanczos_camera(full, M, sx[inside], sy[inside], kx, ky)
        got = out[(iy - 0.5).astype(int), (ix - 0.5).astype(int)][inside]
        err = float(np.abs(got - ref).max())
        assert err < 3e-4 and inside.sum() > 1000, (name, err, kx, ky)
        best[name] = dict(max_diff=err, widening=[round(kx, 4), round(ky, 4)])
    ok(f'{path}: Best (Lanczos-3, widened by the downscale, anti-ringing): equals an independent implementation', **best)


def quality(ok):
    """Best against Fast on test charts: a zone plate through a downscale (aliasing), an edge through an enlargement."""
    # Zone plate: frequency rises with the radius up to 0.45 cycles per source pixel at the corner of a 2400x1600 frame.
    cw, ch = 2400, 1600
    y, x = np.mgrid[0:ch, 0:cw]
    r2 = (x - cw / 2) ** 2 + (y - ch / 2) ** 2
    rmax = math.hypot(cw / 2, ch / 2)
    k = 0.45 / rmax                                        # local frequency = k * r cycles per pixel
    plate = MD.write_dng(SCRATCH / 'ZONE_000001.DNG', cw, ch, scene=0.25 + 0.2 * np.cos(np.pi * k * r2))
    W, H = 960, 640                                        # fit x 0.4: the output can carry up to 0.2 cycles per source pixel
    res = {}
    for name, fast in (('fast_bilinear', True), ('best_lanczos3', False)):
        out = render(plate, (W, H), fast=fast)[..., 1]
        oy, ox = np.mgrid[0:H, 0:W] + 0.5
        f = k * np.hypot(ox / 0.4 - cw / 2, oy / 0.4 - ch / 2)             # source frequency shown at each output pixel
        alias = out[(f > 0.26) & (f < 0.42)]               # beyond what the output can show: should be flat grey
        keep = out[(f > 0.04) & (f < 0.10)]                # well inside: should keep its contrast
        res[name] = dict(aliasing_rms=float(alias.std()), kept_contrast_rms=float(keep.std()))
    ideal = float(np.std(0.2 * np.cos(np.linspace(0, 200 * np.pi, 100001))))
    for v in res.values():
        v['aliasing_percent_of_full_contrast'] = round(100 * v.pop('aliasing_rms') / ideal, 1)
        v['kept_contrast_percent'] = round(100 * v.pop('kept_contrast_rms') / ideal, 1)
    assert res['best_lanczos3']['aliasing_percent_of_full_contrast'] < 0.25 * res['fast_bilinear']['aliasing_percent_of_full_contrast'], res
    assert res['best_lanczos3']['kept_contrast_percent'] > 80, res
    ok('zone plate through a x0.4 fit: aliasing beyond the output\'s limit, and contrast kept inside it', **res)
    # A slanted edge enlarged 4 times: width of the 10-90 % rise across the edge, and overshoot.
    cw, ch = 1200, 800
    y, x = np.mgrid[0:ch, 0:cw]
    d = (x - cw / 2) * math.cos(math.radians(5)) + (y - ch / 2) * math.sin(math.radians(5))
    edge = MD.write_dng(SCRATCH / 'EDGE_000001.DNG', cw, ch, scene=0.1 + 0.3 * (d > 0))   # a hard step, 5 degrees off vertical
    zres = {}
    for name, fast in (('fast_bilinear', True), ('best_lanczos3', False)):
        out = render(edge, (cw, ch), dict(zx=4.0, zy=4.0), fast=fast)[..., 1]
        row = out[ch // 2 - 100:ch // 2 + 100, :].mean(0) if False else out[ch // 2]
        lo, hi = np.median(row[:200]), np.median(row[-200:])
        n = (row - lo) / (hi - lo)
        def crossing(level):                                 # sub-pixel position where the row first passes the level
            i = int(np.argmax(n > level))
            return i - 1 + (level - n[i - 1]) / (n[i] - n[i - 1])
        zres[name] = dict(rise_10_90_output_px=round(float(crossing(0.9) - crossing(0.1)), 2),
                          overshoot_percent=round(100 * float(max(n.max() - 1, -n.min())), 2))
    assert zres['best_lanczos3']['rise_10_90_output_px'] < zres['fast_bilinear']['rise_10_90_output_px'], zres
    assert zres['best_lanczos3']['overshoot_percent'] < 5.0, zres    # the demosaiced edge itself overshoots 2.5 %; Lanczos-3 without the limit would add about 9 %
    ok('edge enlarged 4 times: 10-90 % rise in output pixels and overshoot (anti-ringing)', **zres)


def stabilised(ok, dng, cw, ch):
    """Stabilisation and Transform in the same pass: output -> transform -> fit -> stabilisation -> recorded frame."""
    import test_gyro as TG
    g, _ = TG.pulse_clip(1, 1, frames=30, at=0, counts=0, length=1)
    t = np.arange(len(g)) / TG.RATE
    g[:, 1] = np.round(1500 * np.sin(2 * np.pi * 1.5 * t))
    g[:, 0] = np.round(900 * np.sin(2 * np.pi * 2.2 * t))
    fpg = TG.write_fpg(SCRATCH / 'syn.FPG', g, 30, raster=(cw + 16, ch + 12), active=(8, 6, cw, ch))
    full = render(dng, (cw, ch))
    W, H = 1080, 720
    xf = dict(zx=1.25, zy=1.25, rot=8.0, px=15.0)
    out = render(dng, (W, H), xf, extra=dict(stabEnable=1, gyroFile=str(fpg)), frame=0, fast=True)
    rng = np.random.default_rng(7)
    ix, iy = rng.integers(100, W - 100, 300) + 0.5, rng.integers(100, H - 100, 300) + 0.5
    sx, sy, _ = to_crop(ix, iy, cw, ch, W, H, 'fit', xf)
    j = TG.gyro(fpg, 'frame=0', points=list(zip(sx + 8, sy + 6)))      # crop -> raster, then the stabilisation's map
    src = np.array([[q[2] - 8, q[3] - 6] for q in j['points']])
    ref = bilinear(full, src[:, 0], src[:, 1])
    got = out[(iy - 0.5).astype(int), (ix - 0.5).astype(int)]
    err = float(np.abs(got - ref).max())
    assert j['on'] == 1 and j['zoom'] > 1.0 and err < 5e-4, (err, j['zoom'])
    # The same chain with the Best kernel (its widening follows the whole chain at the frame centre).
    out = render(dng, (W, H), xf, extra=dict(stabEnable=1, gyroFile=str(fpg)), frame=0)
    cpts = [(W / 2, H / 2), (W / 2 + 1, H / 2), (W / 2, H / 2 + 1)]
    cs = to_crop(np.array([q[0] for q in cpts]), np.array([q[1] for q in cpts]), cw, ch, W, H, 'fit', xf)
    jc = np.array([[q[2], q[3]] for q in TG.gyro(fpg, 'frame=0', points=list(zip(cs[0] + 8, cs[1] + 6)))['points']])
    kx, ky = float(np.hypot(*(jc[1:, 0] - jc[0, 0]))), float(np.hypot(*(jc[1:, 1] - jc[0, 1])))
    kx, ky = (kx if kx > 1.001 else 1.0), (ky if ky > 1.001 else 1.0)
    ref = lanczos_camera(full, camera_matrix(dng), src[:, 0], src[:, 1], kx, ky)
    got = out[(iy - 0.5).astype(int), (ix - 0.5).astype(int)]
    err_best = float(np.abs(got - ref).max())
    assert err_best < 5e-4, (err_best, kx, ky)
    ok('stabilisation (auto zoom included) and Transform resampled together, once, from the full-resolution frame',
       auto_zoom=j['zoom'], max_diff_fast=err, max_diff_best=err_best, widening=[round(kx, 4), round(ky, 4)], points=len(ix))


def params(ok, dng, cw, ch):
    host = OFXHost(PLUGIN, canvas=(960, 540))
    host.set(**STAB_TEST)
    names = ['fitMode', 'xfZoomX', 'xfZoomY', 'xfZoomLink', 'xfPosX', 'xfPosY', 'xfRotation', 'xfAnchorX', 'xfAnchorY', 'xfPitch', 'xfYaw', 'xfFlipH', 'xfFlipV']
    assert all(n in host.params() for n in names)
    assert host.choices('fitMode') == ['Scale to Fit', 'Fit Width', 'Fit Height']
    props = {n: host.obj(host.obj(host.params()[n])['props'])['values'] for n in names}
    assert [props[n].get('OfxParamPropParent', [None])[0] for n in names] == ['xform'] * 13   # Fit sits in the Transform group since 1.6.0
    defaults = {n: host.get(n) for n in names}
    assert defaults == dict(fitMode=0, xfZoomX=1.0, xfZoomY=1.0, xfZoomLink=1, xfPosX=0.0, xfPosY=0.0, xfRotation=0.0, xfAnchorX=0.0,
                            xfAnchorY=0.0, xfPitch=0.0, xfYaw=0.0, xfFlipH=0, xfFlipV=0), defaults
    order = list(host.params())
    assert order.index('xform') + 1 == order.index('fitMode') < order.index('stab')
    # Linked zoom: Y follows X (in the picture and in the field).
    host.obj(host.obj(host.instance)['props'])['values']['OfxImageEffectPropSrcFilePath'] = [str(dng)]
    host.set(gamma=0, xfZoomX=1.5)
    host.changed('xfZoomX')
    assert host.get('xfZoomY') == 1.5
    st, linked = host.render(source_frame=0)
    host.set(xfZoomLink=0, xfZoomY=1.0)
    st2, unlinked = host.render(source_frame=0)
    host.set(xfZoomY=1.5)
    st3, same = host.render(source_frame=0)
    assert st == st2 == st3 == OK and not np.array_equal(linked.pixels, unlinked.pixels) and np.array_equal(linked.pixels, same.pixels)
    # The hidden Image Fit of older projects still counts while Fit is Scale to Fit; Fit Width / Height take over.
    host.set(xfZoomX=1.0, xfZoomY=1.0, fit=1)                      # Fill
    _, fill = host.render(source_frame=0)
    host.set(fit=0, fitMode=1)                                     # 2160x1440 into 960x540: Fill = Fit Width
    _, width = host.render(source_frame=0)
    host.set(fitMode=2, fit=1)                                     # Fit Height wins over the stored Fill
    _, height = host.render(source_frame=0)
    host.set(fitMode=0, fit=0)
    _, plain = host.render(source_frame=0)
    host.close()
    assert np.array_equal(fill.pixels, width.pixels) and not np.array_equal(height.pixels, fill.pixels) and np.array_equal(height.pixels, plain.pixels)
    ok("controls declared like Resolve's Transform (Fit above the group, neutral defaults, Link Zoom), older Image Fit still honoured")


def reference_build(ok):
    ref = os.environ.get('SFP_REFERENCE_OFX')
    if not ref:
        print('SKIP neutral Transform = 1.4.2 bit for bit - set SFP_REFERENCE_OFX to a 1.4.2 SigmaFpRaw.ofx')
        return
    sizes = ((3840, 2560), (4320, 2160), (3000, 2000))
    hashes = {}
    for cw, ch in sizes:
        dng = MD.write_dng(SCRATCH / f'REF_{cw}x{ch}_000001.DNG', cw, ch)
        for canvas in ((3840, 2160), (1920, 1080), (4096, 2160)):
            for extra in ({}, dict(sharpness=40.0, fit=1, gamma=3)):
                a = render(dng, canvas, extra=extra, fast=True)
                b = render(dng, canvas, plugin=ref, extra=extra)
                assert np.array_equal(a, b), (cw, ch, canvas, extra)
                hashes[f'{cw}x{ch} -> {canvas[0]}x{canvas[1]}{" sharp+fill" if extra else ""}'] = hashlib.sha256(a.tobytes()).hexdigest()[:12]
        # Best: unchanged where the fit is 1:1 (nothing is resampled), different where the picture is scaled.
        same = render(dng, (cw, ch))
        assert np.array_equal(same, render(dng, (cw, ch), plugin=ref))
        scaled = render(dng, (1920, 1080))
        assert not np.array_equal(scaled, render(dng, (1920, 1080), plugin=ref))
    ok('neutral Transform: Fast is bit for bit the 1.4.2 picture; Best is too when the fit is 1:1 and differs where the picture is scaled',
       cases=len(hashes))


if __name__ == '__main__':
    main()
