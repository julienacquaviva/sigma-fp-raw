"""Integration test of the built SigmaFpRaw.ofx in an independent OFX 1.4 host (CPU image
path, so the plug-in uses its own CUDA context and copies the result back).
Uses real Sigma fp frames (read-only). usage: python tests/test_ofx.py [CLIP_DIR]"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
os.environ.setdefault('SFP_GYRO_CACHE_DIR', 'off')        # no gyro cache files from test runs
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from ofx_host import OFXHost, OK  # noqa: E402

PLUGIN = ROOT / 'dist/SigmaFpRaw.ofx.bundle/Contents/Win64/SigmaFpRaw.ofx'
CLIP = Path(sys.argv[1] if len(sys.argv) > 1 else 'F:/CINEMA/A001_065')


def cli_reference(path, w, h, *opts):
    out = Path(tempfile.gettempdir()) / f'sfp_ref_{Path(path).stem}.tif'
    subprocess.run([str(ROOT / 'build/sfp_cli.exe'), 'develop', str(path), str(out), f'width={w}', f'height={h}', *opts],
                   check=True, capture_output=True)
    return tifffile.imread(out).astype(np.float64) / 65535.0


def main():
    frames = sorted(CLIP.glob('*.DNG'))
    first = frames[0]
    report = []

    def ok(name, **info):
        report.append(dict(name=name, passed=True, **info))
        print('PASS', name, info if info else '')

    host = OFXHost(PLUGIN, canvas=(3008, 2000))
    params = set(host.params())
    raw_controls = {'decodeQuality', 'whiteBalance', 'colorSpace', 'gamma', 'colorTemp', 'tint', 'exposure', 'sharpness',
                    'highlights', 'shadows', 'colorBoost', 'saturation', 'midtones', 'lift', 'gain', 'contrast',
                    'highlightRecovery', 'gamutMapping', 'preToneCurve', 'softClip'}
    assert raw_controls <= params, raw_controls - params
    assert {'rowPhase', 'deZigzag', 'fit'} <= params and not {'rowPhaseMode', 'deZigzagMode'} & params
    assert host.choices('whiteBalance') == ['As Shot', 'Daylight', 'Cloudy', 'Shade', 'Tungsten', 'Fluorescent', 'Flash', 'Custom']
    ok('load, describe, instance; Camera RAW + Sigma fp controls declared', controls=len(params))

    # Controls taken out of the panel in v1.3: still parameters (old projects load), hidden, at their defaults.
    def prop(name, key):
        return host.obj(host.obj(host.params()[name])['props'])['values'].get(key, [None])[0]
    hidden = {'highlightRecovery': 0, 'gamutMapping': 0, 'preToneCurve': 0, 'softClip': 0, 'frameMode': 0, 'anchor': 0.0,
              'rowPhase': -0.125, 'deZigzag': 80.0, 'fit': 0, 'sourceFile': ''}
    for name, default in hidden.items():
        assert prop(name, 'OfxParamPropSecret') == 1 and prop(name, 'OfxParamPropDefault') == default and host.get(name) == default, name
    assert host.choices('frameMode')[0] == 'Resolve Source Frame'
    assert host.choices('fit')[0] == 'Scale to Fit'
    # The "Sigma fp" and "Advanced" groups are gone (v1.3.2): their controls are hidden and belong to no group.
    assert 'sigma' not in params and 'source' not in params
    assert all(prop(n, 'OfxParamPropParent') is None for n in ('rowPhase', 'deZigzag', 'fit', 'sourceFile', 'frameMode', 'anchor'))
    groups = sorted(n for n in params if prop(n, 'OfxParamPropType') == 'OfxParamTypeGroup')
    assert groups == ['raw', 'stab'], groups
    visible = sorted(n for n in params if not prop(n, 'OfxParamPropSecret') and prop(n, 'OfxParamPropParent') not in ('raw', 'stab') and n not in groups)
    assert visible == ['Controls', 'info'], visible
    shown = sorted(n for n in params if not prop(n, 'OfxParamPropSecret') and prop(n, 'OfxParamPropParent') == 'raw')
    assert 'highlightRecovery' not in shown and 'exposure' in shown and len(shown) == 16, shown
    ok('hidden: Highlight Recovery, Gamut Mapping, Pre Tone Curve, Soft Clip (all off), Frame Mapping (Resolve Source Frame), '
       'Timeline Anchor (0), Row Phase (-0.125), Edge Anti-aliasing (80), Image Fit (Scale to Fit), First DNG File (empty); '
       'no Sigma fp or Advanced group', hidden_defaults=hidden)

    # No source at all -> clear error, no crash.
    status, _ = host.render(source_frame=0)
    assert status != OK and any('could not be identified' in m['text'] for m in host.messages[-2:]), host.messages[-2:]
    ok('no source: explicit error')

    # Automatic Resolve source path + source frame (0-based offset from the first file).
    props = host.obj(host.obj(host.instance)['props'])['values']
    props['OfxImageEffectPropSrcFilePath'] = [str(first)]
    for idx in (100, 101):                       # A001_065: frame 101 LJ92-compressed, 102 uncompressed
        status, out = host.render(source_frame=idx)
        assert status == OK, host.messages[-3:]
        img = out.pixels[::-1, :, :3].astype(np.float64)   # OFX bottom-up -> top-down
        ref = cli_reference(frames[idx], 3008, 2000)
        diff = np.abs(np.clip(img, 0, 1) - ref).max()
        assert diff <= 1.0 / 65535 + 1e-6, diff
        assert np.all(out.pixels[..., 3] == 1.0)
        ok(f'Resolve source frame {idx} -> {frames[idx].name}: matches CLI develop', max_diff=float(diff))

    # Padded negative stride and a partial render window (bounds) equal the full render.
    status, full = host.render(source_frame=100)
    status, neg = host.render(source_frame=100, negative_stride=True)
    assert status == OK and np.array_equal(neg.pixels, full.pixels)
    status, part = host.render(source_frame=100, bounds=(1000, 500, 1600, 900))
    assert status == OK and np.array_equal(part.pixels, full.pixels[500:900, 1000:1600])
    ok('negative row stride and sub-bounds match the full frame')

    # Render scale 0.5 (Resolve proxy/playback) keeps framing.
    status, half = host.render(source_frame=100, scale=(0.5, 0.5))
    assert status == OK and half.pixels.shape == (1000, 1504, 4)
    small = full.pixels[:, :, :3].reshape(1000, 2, 1504, 2, 3).mean(axis=(1, 3))
    corr = np.corrcoef(small.ravel(), half.pixels[:, :, :3].ravel())[0, 1]
    assert corr > 0.99, corr
    ok('render scale 0.5', correlation=float(corr))

    # Missing frame -> error.
    status, _ = host.render(source_frame=999999)
    assert status != OK and any('Cannot open' in m['text'] for m in host.messages[-2:]), host.messages[-2:]
    ok('missing frame: explicit error')

    # Timeline anchor with an explicit first file.
    host.set(sourceFile=str(first), frameMode=1, anchor=50.0)
    status, anch = host.render(time=150.0, source_frame=None)
    assert status == OK and np.array_equal(anch.pixels, full.pixels)
    host.set(sourceFile='', frameMode=0, anchor=0.0)
    ok('timeline anchor mode (time 150, anchor 50 -> file 101)')

    # The way out when the host gives neither the clip nor the source frame (e.g. the Fusion page):
    # First DNG File alone is enough, the frame then follows the effect time.
    props.pop('OfxImageEffectPropSrcFilePath')
    status, _ = host.render(time=100.0, source_frame=None)
    assert status != OK and any('could not be identified' in m['text'] for m in host.messages[-2:]), host.messages[-2:]
    host.set(sourceFile=str(first))
    status, alone = host.render(time=100.0, source_frame=None)
    assert status == OK and np.array_equal(alone.pixels, full.pixels)
    host.set(sourceFile='')
    props['OfxImageEffectPropSrcFilePath'] = [str(first)]
    ok('no clip and no source frame from the host: clear error; a project with First DNG File stored still renders (time 100 -> file 101)')

    # A hidden checkbox still does what an older project stored for it.
    host.set(highlightRecovery=1, gamutMapping=1, preToneCurve=1, softClip=1)
    _, hr = host.render(source_frame=100)
    host.set(highlightRecovery=0, gamutMapping=0, preToneCurve=0, softClip=0)
    _, back = host.render(source_frame=100)
    assert not np.array_equal(hr.pixels, full.pixels) and np.array_equal(back.pixels, full.pixels)
    ok('hidden controls keep working for projects that set them (the four checkboxes on change the picture, off = as before)')

    # Controls change the image; Exposure +1 doubles linear output.
    host.set(gamma=0)
    _, lin0 = host.render(source_frame=101)
    host.set(exposure=1.0)
    _, lin1 = host.render(source_frame=101)
    host.set(exposure=0.0, gamma=3)
    m = (lin0.pixels[..., 1] > 0.01) & (lin0.pixels[..., 1] < 0.2)
    ratio = float(np.median(lin1.pixels[..., 1][m] / lin0.pixels[..., 1][m]))
    assert abs(ratio - 2.0) < 1e-3, ratio
    ok('exposure +1 stop = x2 linear', ratio=ratio)

    host.close()
    # UHD timeline: 3008x2000 scaled to fit 3840x2160 -> pillarbox.
    uhd = OFXHost(PLUGIN, canvas=(3840, 2160))
    uhd.obj(uhd.obj(uhd.instance)['props'])['values']['OfxImageEffectPropSrcFilePath'] = [str(first)]
    status, u = uhd.render(source_frame=100)
    assert status == OK
    left = int(round((3840 - 3008 * 2160 / 2000) / 2))
    assert np.all(u.pixels[:, :left - 1, :3] == 0) and u.pixels[:, left + 2:3840 - left - 2, :3].mean() > 0.05
    uhd.close()
    ok('UHD timeline: scale to fit with pillarbox', pillar=left)

    (ROOT / 'tests/ofx_report.json').write_text(json.dumps({'plugin': str(PLUGIN), 'tests': report}, indent=2) + '\n')
    print(f'ALL PASS ({len(report)})')


if __name__ == '__main__':
    main()
