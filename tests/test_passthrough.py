"""Develop RAW off: the host's own picture of the clip goes through the plug-in untouched in
colour; only the Transform, the lens distortion and the stabilisation move it.
Runs without camera files (a synthetic DNG gives the clip's size; the picture is made here).
usage: python tests/test_passthrough.py        (GPU kernels, then the same on the processor path)"""
import json, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault('SFP_GYRO_CACHE_DIR', 'off')
os.environ.setdefault('SFP_ADOBE_PROFILES', 'off')         # no installed lens profile may step in
sys.path.insert(0, str(HERE))
from ofx_host import OFXHost, OK, ImageBuffer  # noqa: E402
import make_test_dng as MD  # noqa: E402

PLUGIN = ROOT / 'dist/SigmaFpRaw.ofx.bundle/Contents/Win64/SigmaFpRaw.ofx'
SCRATCH = HERE / 'scratch/passthrough'
CPU_PART = os.environ.get('SFP_CPU') == '1'


def picture(w, h, pw, ph, seed=3):
    """A host picture w x h: the clip's frame pw x ph centred in it (smooth colours + fine detail), black around."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:ph, 0:pw].astype(np.float32)
    pic = np.stack([0.2 + 0.6 * x / pw, 0.3 + 0.5 * y / ph, 0.5 + 0.3 * np.sin(x / 37.0) * np.cos(y / 23.0)], axis=-1)
    pic += rng.normal(0, 0.01, pic.shape).astype(np.float32)
    img = np.zeros((h, w, 4), np.float32)
    img[..., 3] = 1
    x0, y0 = (w - pw) // 2, (h - ph) // 2
    img[y0:y0 + ph, x0:x0 + pw, :3] = pic
    return img, (x0, y0)


def give_source(host, img, negative_stride=False):
    """Hands the top-down picture to the host as the Source clip's image (OFX rows are bottom-up)."""
    h, w = img.shape[:2]
    buf = ImageBuffer((0, 0, w, h), negative_stride=negative_stride)
    buf.pixels[:] = img[::-1]
    props = host.props({
        'OfxImagePropData': [buf.pointer],
        'OfxImagePropBounds': [0, 0, w, h],
        'OfxImagePropRegionOfDefinition': [0, 0, w, h],
        'OfxImagePropRowBytes': [buf.row_bytes],
        'OfxImageEffectPropComponents': ['OfxImageComponentRGBA'],
        'OfxImageEffectPropPixelDepth': ['OfxBitDepthFloat'],
    })
    host.obj(host.obj(host.instance)['clips']['Source'])['image'] = props
    return buf


def main():
    report = []

    def ok(name, **info):
        report.append(dict(name=name, passed=True, **info))
        print('PASS', name, info if info else '')

    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    cw, ch = 3000, 2000                                   # the clip's frame (3:2)
    dng = MD.write_dng(SCRATCH / 'SYN_PT_000001.DNG', cw, ch)
    W, H = 1620, 1080                                     # the node's frame is the clip's own frame (the host places it in the timeline afterwards)
    pw, ph = 1620, 1080
    img, (x0, y0) = picture(W, H, pw, ph)

    host = OFXHost(PLUGIN, canvas=(W, H))
    host.obj(host.obj(host.instance)['props'])['values']['OfxImageEffectPropSrcFilePath'] = [str(dng)]
    host.changed('Source')
    assert host.get('developRaw') == 0 and host.get('stabEnable') == 0 and (host.get('lensShading'), host.get('lensDistortion')) == (0, 0)
    host.set(developRaw=1)
    keep = give_source(host, img)

    # Developing on: the source picture is not asked for, and the output is the developed frame.
    before = host.source_image_requests
    status, dev = host.render(source_frame=0)
    assert status == OK and host.source_image_requests == before
    assert not np.allclose(dev.pixels[::-1, :, :3], img[..., :3], atol=0.05)

    # Develop RAW off, everything neutral: the host's picture comes out bit for bit.
    host.set(developRaw=0, lensDistortion=0)
    status, out = host.render(source_frame=0)
    assert status == OK, host.messages[-2:]
    got = out.pixels[::-1]
    assert host.source_image_requests == before + 1
    assert np.array_equal(got[..., :3], img[..., :3]) and np.all(got[..., 3] == 1)
    ok('Develop RAW off, neutral: the output is the host picture, bit for bit')

    # The same with a padded, negative-stride source image and a partial render window.
    keep = give_source(host, img, negative_stride=True)
    status, neg = host.render(source_frame=0, negative_stride=True)
    assert status == OK and np.array_equal(neg.pixels[::-1][..., :3], img[..., :3])
    status, part = host.render(source_frame=0, bounds=(400, 200, 1000, 700))
    assert status == OK and np.array_equal(part.pixels, out.pixels[200:700, 400:1000])
    keep = give_source(host, img)
    ok('negative row stride and a partial render window')

    # Transform: zoom 2 about the centre = each source pixel of the middle twice as large (bilinear resampling for an exact reference).
    host.set(xfZoomX=2.0, xfZoomY=2.0, resampling=1)
    status, z = host.render(source_frame=0)
    zoomed = z.pixels[::-1][..., :3]
    yy, xx = np.mgrid[100:980:7, 100:W - 100:7]
    sx, sy = (xx + 0.5 - W / 2) / 2 + W / 2 - 0.5, (yy + 0.5 - H / 2) / 2 + H / 2 - 0.5
    fx, fy = (sx - np.floor(sx))[..., None], (sy - np.floor(sy))[..., None]
    ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
    ref = img[iy, ix, :3] * (1 - fx) * (1 - fy) + img[iy, ix + 1, :3] * fx * (1 - fy) + img[iy + 1, ix, :3] * (1 - fx) * fy + img[iy + 1, ix + 1, :3] * fx * fy
    err = float(np.abs(zoomed[yy, xx] - ref).max())
    assert status == OK and err < 2e-4, err
    host.set(xfZoomX=1.0, xfZoomY=1.0)
    ok('Transform zoom 2 on the host picture', max_error=err)

    # Lens distortion from a profile file (Lensfun poly3, calibrated on full frame): a point of the
    # output is read at  centre + offset x (1 - k1 + k1 r^2),  r = 1 at half the shorter side of the
    # sensor (12 mm of 23.93 mm = 2016 sensor rows... in picture pixels: ph x 12 / (4032 x 35.9 / 6000)).
    k1 = -0.04
    lf = SCRATCH / 'lens.xml'
    lf.write_text(f'<lensdatabase version="1"><lens><maker>T</maker><model>Test 28mm F2</model><cropfactor>1</cropfactor>'
                  f'<calibration><distortion model="poly3" focal="28" k1="{k1}"/></calibration></lens></lensdatabase>')
    host.set(lensDistortion=1, lensDistortionFile=str(lf))
    host.changed('lensDistortionFile')
    assert 'distortion: file, Lensfun' in host.get('infoCorrection') and 'vignette: off' in host.get('infoCorrection'), host.get('infoCorrection')
    status, d = host.render(source_frame=0)
    dist = d.pixels[::-1][..., :3]
    radius = 12.0 / (35.9 / 6000) * (pw / 6048.0)          # 12 mm in sensor pixels, then in picture pixels (the frame shows the whole sensor width)
    yy, xx = np.mgrid[60:1020:9, 60:W - 60:9]
    dx, dy = (xx + 0.5 - W / 2) / radius, (yy + 0.5 - H / 2) / radius
    f = 1 - k1 + k1 * (dx * dx + dy * dy)
    sx, sy = W / 2 + dx * f * radius - 0.5, H / 2 + dy * f * radius - 0.5
    fx, fy = (sx - np.floor(sx))[..., None], (sy - np.floor(sy))[..., None]
    ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
    inside = (ix >= x0) & (ix + 1 < x0 + pw) & (iy >= 0) & (iy + 1 < H)
    ref = img[iy, np.clip(ix, 0, W - 2), :3] * (1 - fx) * (1 - fy) + img[iy, np.clip(ix, 0, W - 2) + 1, :3] * fx * (1 - fy) + \
        img[np.clip(iy, 0, H - 2) + 1, np.clip(ix, 0, W - 2), :3] * (1 - fx) * fy + img[np.clip(iy, 0, H - 2) + 1, np.clip(ix, 0, W - 2) + 1, :3] * fx * fy
    err = float(np.abs(dist[yy, xx] - ref)[inside].max())
    moved = float(np.abs(dist - img[..., :3]).mean())
    assert status == OK and err < 3e-3 and moved > 1e-3, (err, moved)
    host.set(lensDistortion=0, lensDistortionFile='', resampling=0)
    ok('lens distortion from a profile file on the host picture', max_error=err, mean_change=moved)

    # Vignette on the host picture: a profile file (Lensfun "pa": falloff 1 + k1 r^2 + k2 r^4 + k3 r^6, r = 1 at the corner of a
    # full-frame sensor, 21.63 mm) is removed in linear light: out = encode(decode(in) / falloff), for every Resolve Gamma.
    kv = (-0.45, 0.12, -0.03)
    vf = SCRATCH / 'vig.xml'
    vf.write_text(f'<lensdatabase version="1"><lens><maker>T</maker><model>Test 28mm F2</model><cropfactor>1</cropfactor><calibration>'
                  f'<vignetting model="pa" focal="28" aperture="2" distance="10" k1="{kv[0]}" k2="{kv[1]}" k3="{kv[2]}"/></calibration></lens></lensdatabase>')
    host.set(lensShading=1, lensShadingFile=str(vf))
    host.changed('lensShadingFile')
    assert "brightness only, on Resolve's picture" in host.get('infoCorrection'), host.get('infoCorrection')
    yy, xx = np.mgrid[0:H, 0:W]
    mm = (35.9 / 6000) * (6048.0 / pw)                    # millimetres on the sensor per picture pixel
    r2 = (((xx + 0.5 - W / 2) * mm) ** 2 + ((yy + 0.5 - H / 2) * mm) ** 2) / (0.5 * np.hypot(36.0, 24.0)) ** 2
    gain = (1.0 / (1 + kv[0] * r2 + kv[1] * r2 ** 2 + kv[2] * r2 ** 3))[..., None]

    def bmd(A, B, C, cut):
        lc = np.exp((cut - C) / A) - B
        return (lambda e: np.where(e >= cut, np.exp((e - C) / A) - B, lc + (e - cut) * (lc + B) / A),
                lambda v: np.where(v >= lc, A * np.log(np.maximum(v, lc) + B) + C, cut + (v - lc) * A / (lc + B)))
    curves = {
        0: (lambda e: e, lambda v: v),
        2: (lambda e: e ** 2.4, lambda v: v ** (1 / 2.4)),
        3: (lambda e: e ** 2.6, lambda v: v ** (1 / 2.6)),
        4: (lambda e: np.where(e < 0.081, e / 4.5, ((e + 0.099) / 1.099) ** (1 / 0.45)), lambda v: np.where(v < 0.018, 4.5 * v, 1.099 * v ** 0.45 - 0.099)),
        5: (lambda e: np.where(e <= 0.04045, e / 12.92, ((e + 0.055) / 1.055) ** 2.4), lambda v: np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)),
        6: bmd(0.18644097, 0.03251850, 0.67230670, 0.060236),
        7: bmd(0.29529631, 0.07974455, 0.78163686, 0.052812),
        8: bmd(0.15753964, 0.02359390, 0.66140084, 0.096414),
        9: (lambda e: np.where(e <= 0.00262409 * 10.44426855, e / 10.44426855, 2.0 ** (e / 0.07329248 - 7.0) - 0.0075),
            lambda v: np.where(v <= 0.00262409, v * 10.44426855, (np.log2(v + 0.0075) + 7.0) * 0.07329248)),
        10: (lambda e: np.where(e <= 0.155251141552511, (e - 0.0729055341958355) / 10.5402377416545, 2.0 ** (e * 17.52 - 9.72)),
             lambda v: np.where(v <= 0.0078125, 10.5402377416545 * v + 0.0729055341958355, (np.log2(np.maximum(v, 1e-9)) + 9.72) / 17.52)),
    }
    area = (slice(None), slice(x0, x0 + pw))
    src = img[..., :3].astype(np.float64)
    errs = {}
    for gid, (dec, enc) in curves.items():
        host.set(sourceGamma=gid)
        status, v = host.render(source_frame=0)
        assert status == OK, host.messages[-2:]
        want = enc(dec(np.maximum(src, 0)) * gain)
        errs[host.choices('sourceGamma')[gid]] = float(np.abs(v.pixels[::-1][..., :3] - want)[area].max())
        assert errs[host.choices('sourceGamma')[gid]] < 4e-3, errs
    corner = float(gain[0, x0, 0])
    # Blackmagic Design Film against Resolve's own table, when Resolve is installed here.
    table = Path(os.environ.get('ProgramData', '')) / 'Blackmagic Design/DaVinci Resolve/Support/LUT/VFX IO/BMDFilm to Linear.cube'
    table_error = None
    if table.exists():
        lin = np.array([float(line.split()[0]) for line in table.read_text().splitlines() if line and line[0] in '-0123456789'])
        code = np.linspace(0, 1, len(lin))
        table_error = float(np.abs(curves[6][0](code) - lin)[lin < lin.max() - 1e-9].max())
        assert table_error < 2e-5, table_error
    host.set(lensShading=0, lensShadingFile='', sourceGamma=4)
    status, back = host.render(source_frame=0)
    assert np.array_equal(back.pixels[::-1][..., :3], img[..., :3])
    ok('vignette removed from the host picture in linear light, every Resolve Gamma', corner_gain=corner, max_error=errs,
       blackmagic_film_formula_against_resolve_table=table_error)

    # Stabilisation: a synthetic gyro file next to the clip; the result moves the picture, keeps its colours, and
    # switching it off again returns the host picture.
    try:
        import test_gyro as TG
        n = 2500
        g = np.zeros((n, 3), np.int16)
        t = np.arange(n) / TG.RATE
        g[:, 1] = np.round(1500 * np.sin(2 * np.pi * 1.5 * t))
        g[:, 0] = np.round(900 * np.sin(2 * np.pi * 2.2 * t))
        fpg = TG.write_fpg(SCRATCH / 'syn.FPG', g, 20, raster=(cw + 16, ch + 12), active=(8, 6, cw, ch))
        host.set(stabEnable=1, stabSmoothness=0.5, stabZoomMode=0, gyroFile=str(fpg))
        host.changed('gyroFile')
        assert host.get('stabStatus').startswith('On'), host.get('stabStatus')
        status, st = host.render(source_frame=0)
        assert status == OK, host.messages[-2:]
        stab = st.pixels[::-1][..., :3]
        change = float(np.abs(stab - img[..., :3])[:, x0 + 50:x0 + pw - 50].mean())
        assert status == OK and change > 1e-3 and float(stab.max()) <= float(img[..., :3].max()) + 0.05 and float(stab[:, x0 + 100:x0 + pw - 100].min()) > 0.05, change
        # Against the developed, stabilised frame: the same geometry (the pillarbox edges fall on the same columns).
        host.set(stabEnable=0)
        status, back = host.render(source_frame=0)
        assert np.array_equal(back.pixels[::-1][..., :3], img[..., :3])
        ok('stabilisation on the host picture (moves it, no borders, colours kept); off = the host picture again', mean_change=change)
    except ImportError as e:
        print('SKIP stabilisation -', e)

    host.close()
    # A file whose crop tags give another shape than the host shows (seen: a 2:1 picture with 16:9 tags): the host's
    # picture still goes through whole, nothing is cut to the tags.
    wide = OFXHost(PLUGIN, canvas=(2000, 1000))
    wide.obj(wide.obj(wide.instance)['props'])['values']['OfxImageEffectPropSrcFilePath'] = [str(dng)]
    wide.changed('Source')
    wimg, _ = picture(2000, 1000, 2000, 1000, seed=5)
    keep2 = give_source(wide, wimg)
    status, wout = wide.render(source_frame=0)
    assert status == OK and np.array_equal(wout.pixels[::-1][..., :3], wimg[..., :3])
    wide.close()
    ok('a node frame of another shape than the file says (2:1 frame, 3:2 file): the host picture comes out whole, bit for bit')

    if not CPU_PART:
        env = dict(os.environ, SFP_CPU='1')
        r = subprocess.run([sys.executable, str(Path(__file__).resolve())], env=env, capture_output=True, text=True)
        assert r.returncode == 0 and 'ALL PASS' in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
        ok('the same tests on the processor path (SFP_CPU=1)', result=r.stdout.strip().splitlines()[-1])
        (ROOT / 'tests/passthrough_report.json').write_text(json.dumps({'plugin': str(PLUGIN), 'tests': report}, indent=2) + '\n')
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f'ALL PASS ({len(report)})')


if __name__ == '__main__':
    main()
