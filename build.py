"""Build Sigma fp RAW: the OFX bundle and the command-line tool, for Windows, macOS and Linux.

usage: python build.py [--targets win64,linux64,macos | all] [--package] [--install]

  (no option)   builds for this machine into dist/SigmaFpRaw.ofx.bundle and build/sfp_cli
  --targets     cross-compiles with zig (any host can build every target)
  --package     also writes dist/SigmaFpRaw-<version>-all-platforms.zip (bundle + install scripts)
  --install     copies the bundle into this machine's OFX plug-in folder (needs admin / sudo rights)

Compiler: zig (clang) for every target; on a Mac the macOS binary is built with Apple's clang
as a universal bundle. zig is taken from $ZIG, the toolchain folder next to this project, the
PATH, or the `ziglang` Python package (pip install ziglang).
The CUDA kernels are compiled to PTX with NVRTC when it is installed (it comes with DaVinci
Resolve on Windows); otherwise the PTX kept in src/generated is used.
"""
import argparse, hashlib, json, os, shutil, stat, struct, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = '1.9.3'
PLUGIN_ID = 'com.sigmafpmods.raw'
CORE = ['dng.cpp', 'lj92.cpp', 'color.cpp', 'cuda_api.cpp', 'frame_cache.cpp', 'develop.cpp', 'kernels_cpu.cpp',
        'gyro.cpp', 'gyro_inframe.cpp', 'platform.cpp', 'lens_profile.cpp']
INCLUDES = ['src', 'src/generated', 'third_party/openfx/include']
COMMON = ['-std=c++17', '-O3', '-DNDEBUG', '-Wall', '-Wextra']

# name: (bundle folder, [(zig target, zig cpu)], executable suffix)
TARGETS = {
    'win64': ('Win64', [('x86_64-windows-gnu', 'x86_64_v3')], '.exe'),
    'linux64': ('Linux-x86-64', [('x86_64-linux-gnu.2.17', 'x86_64_v3')], ''),
    # Apple Silicon + Intel. The Intel half stays at x86_64_v2 so that it also runs under Rosetta.
    'macos': ('MacOS', [('aarch64-macos.11.0', 'apple_m1'), ('x86_64-macos.11.0', 'x86_64_v2')], ''),
}
HOST = 'win64' if sys.platform == 'win32' else 'macos' if sys.platform == 'darwin' else 'linux64'


def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def find_zig():
    local = ROOT.parent / 'resolve_dng_plugin/toolchain/zig-x86_64-windows-0.14.1/zig.exe'
    if os.environ.get('ZIG'):
        return [os.environ['ZIG']]
    if local.exists():
        return [local]
    if shutil.which('zig'):
        return [shutil.which('zig')]
    try:
        import ziglang  # noqa: F401
        return [sys.executable, '-m', 'ziglang']
    except ImportError:
        sys.exit('zig not found: pip install ziglang, or set ZIG to the zig executable')


def flags():
    return COMMON + [f'-DSFP_VERSION="{VERSION}"'] + ['-I' + str(ROOT / d) for d in INCLUDES]


def zig_build(zig, target, cpu, sources, out, shared):
    cmd = [*zig, 'c++', *flags(), '-target', target, '-mcpu=' + cpu]
    if 'windows' not in target:
        cmd += ['-fPIC', '-fvisibility=hidden']
    if 'linux' in target:
        cmd += ['-ldl', '-lpthread', '-s']
    if shared:
        cmd.append('-shared')
    out.parent.mkdir(parents=True, exist_ok=True)
    run([*cmd, *sources, '-o', out])
    for junk in out.parent.glob('*'):   # the Windows linker leaves an import library and debug data
        if junk.suffix.lower() in {'.pdb', '.lib'}:
            junk.unlink()


def apple_build(sources, out, shared):
    # With the Metal backend (Objective-C++, only where Apple's compiler and frameworks are).
    sources = [*sources, ROOT / 'src/develop_metal.mm']
    cmd = ['clang++', *flags(), '-DSFP_METAL=1', '-fobjc-arc', '-framework', 'Metal', '-framework', 'Foundation',
           '-arch', 'arm64', '-arch', 'x86_64', '-mmacosx-version-min=11.0', '-fvisibility=hidden']
    if shared:
        cmd.append('-bundle')
    out.parent.mkdir(parents=True, exist_ok=True)
    run([*cmd, *sources, '-o', out])
    run(['codesign', '--force', '--sign', '-', out])


def write_fat(slices, out):
    """Universal (fat) Mach-O from thin ones, as `lipo -create` does."""
    cpu = {b'\x0c\x00\x00\x01': (0x0100000C, 0, 14), b'\x07\x00\x00\x01': (0x01000007, 3, 12)}   # arm64, x86_64
    blobs = [Path(p).read_bytes() for p in slices]
    header = struct.pack('>II', 0xCAFEBABE, len(blobs))
    offset = 4096
    entries, body = b'', b''
    for b in blobs:
        assert b[:4] == b'\xcf\xfa\xed\xfe', 'not a 64-bit Mach-O file'
        cputype, subtype, align = cpu[b[4:8]]
        offset = (offset + (1 << align) - 1) & ~((1 << align) - 1)
        entries += struct.pack('>IIIII', cputype, subtype, offset, len(b), align)
        body += b'\0' * (offset - 4096 - len(body)) + b
        offset += len(b)
    head = header + entries
    Path(out).write_bytes(head + b'\0' * (4096 - len(head)) + body)


def build_target(name, zig, bundle):
    folder, variants, exe = TARGETS[name]
    core = [ROOT / 'src' / n for n in CORE]
    plugin_src = [ROOT / 'src/ofx_plugin.cpp', *core]
    cli_src = [ROOT / 'tools/sfp_cli.cpp', *core]
    binary = bundle / 'Contents' / folder / 'SigmaFpRaw.ofx'
    cli = ROOT / 'build' / ('sfp_cli' + exe if name == HOST else f'{name}/sfp_cli{exe}')
    if name == 'macos' and HOST == 'macos':
        # Built and signed outside the bundle: signing a file inside it would seal the bundle's
        # other files, which are still to be written.
        apple_build(plugin_src, ROOT / 'build/macos/SigmaFpRaw.ofx', True)
        binary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'build/macos/SigmaFpRaw.ofx', binary)
        apple_build(cli_src, cli, False)
    elif len(variants) == 1:
        target, cpu = variants[0]
        zig_build(zig, target, cpu, plugin_src, binary, True)
        zig_build(zig, target, cpu, cli_src, cli, False)
    else:
        thin_plugin, thin_cli = [], []
        for target, cpu in variants:
            tmp = ROOT / 'build' / name / target
            zig_build(zig, target, cpu, plugin_src, tmp / 'SigmaFpRaw.ofx', True)
            zig_build(zig, target, cpu, cli_src, tmp / 'sfp_cli', False)
            thin_plugin.append(tmp / 'SigmaFpRaw.ofx')
            thin_cli.append(tmp / 'sfp_cli')
        binary.parent.mkdir(parents=True, exist_ok=True)
        write_fat(thin_plugin, binary)
        write_fat(thin_cli, cli)
    print(f'{name}: {binary.relative_to(ROOT)} ({binary.stat().st_size} bytes)')
    return binary


INFO_PLIST = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>English</string>
    <key>CFBundleExecutable</key><string>SigmaFpRaw.ofx</string>
    <key>CFBundleIdentifier</key><string>{PLUGIN_ID}</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleName</key><string>SigmaFpRaw</string>
    <key>CFBundlePackageType</key><string>BNDL</string>
    <key>CFBundleShortVersionString</key><string>{VERSION}</string>
    <key>CFBundleVersion</key><string>{VERSION}</string>
    <key>CFBundleSignature</key><string>????</string>
</dict>
</plist>
'''


def ofx_dir():
    if HOST == 'win64':
        return Path(os.environ.get('CommonProgramFiles', r'C:\Program Files\Common Files')) / 'OFX/Plugins'
    return Path('/Library/OFX/Plugins') if HOST == 'macos' else Path('/usr/OFX/Plugins')


def package(bundle):
    """One zip for every system: the bundle folder to copy, and an install script per system."""
    out = ROOT / f'dist/SigmaFpRaw-{VERSION}-all-platforms.zip'
    extras = ['INSTALL.txt', 'Install-Windows.bat', 'Install-macOS.command', 'install-linux.sh']
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        def add(path, arc, executable):
            info = zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3   # Unix: keeps the executable bit on macOS and Linux
            info.external_attr = (stat.S_IFREG | (0o755 if executable else 0o644)) << 16
            data = path.read_bytes()
            if path.suffix in {'.bat', '.txt', '.command', '.sh'}:   # line ends each system expects
                data = data.replace(b'\r\n', b'\n')
                if path.suffix in {'.bat', '.txt'}:
                    data = data.replace(b'\n', b'\r\n')
            z.writestr(info, data)
        for f in sorted(bundle.rglob('*')):
            if f.is_file():
                add(f, 'SigmaFpRaw.ofx.bundle/' + f.relative_to(bundle).as_posix(), f.suffix == '.ofx')
        for n in extras:
            add(ROOT / 'packaging' / n, n, n.endswith(('.command', '.sh')))
        add(ROOT / 'LICENSE', 'LICENSE.txt', False)
    print(f'package: {out.relative_to(ROOT)} ({out.stat().st_size} bytes)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--targets', default=HOST, help='comma-separated: win64, linux64, macos, or all')
    ap.add_argument('--package', action='store_true')
    ap.add_argument('--install', action='store_true')
    a = ap.parse_args()
    names = list(TARGETS) if a.targets == 'all' else a.targets.split(',')
    for n in names:
        if n not in TARGETS:
            sys.exit(f'unknown target {n}')
    run([sys.executable, ROOT / 'tools/make_ptx.py'])
    run([sys.executable, ROOT / 'tools/make_msl.py'])
    zig = None if names == ['macos'] and HOST == 'macos' else find_zig()
    bundle = ROOT / 'dist/SigmaFpRaw.ofx.bundle'
    built = {n: build_target(n, zig, bundle) for n in names}
    (bundle / 'Contents/Resources').mkdir(parents=True, exist_ok=True)
    (bundle / 'Contents/Info.plist').write_text(INFO_PLIST, encoding='utf-8', newline='\n')
    shutil.copy2(ROOT / 'README.md', bundle / 'Contents/Resources/README.md')
    info = {'name': 'Sigma fp RAW', 'version': VERSION, 'plugin_id': PLUGIN_ID,
            'sha256': {TARGETS[n][0]: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in built.items()}}
    (bundle / 'Contents/Resources/build.json').write_text(json.dumps(info, indent=2) + '\n', newline='\n')
    print(json.dumps(info))
    if a.package:
        package(bundle)
    if a.install:
        dest = ofx_dir() / 'SigmaFpRaw.ofx.bundle'
        try:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(bundle, dest)
            print('installed to', dest)
        except PermissionError:
            sys.exit(f'No permission to write {dest}: run again as administrator (sudo on macOS and Linux), '
                     'or use the install script in packaging/.')


if __name__ == '__main__':
    main()
