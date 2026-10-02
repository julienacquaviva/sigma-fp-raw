"""Loads the built plug-in library the way an OpenFX host does and asks it for its plug-in.
usage: python tools/check_plugin.py [path/to/SigmaFpRaw.ofx]   (default: this system's binary in dist/)"""
import ctypes, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = 'Win64' if sys.platform == 'win32' else 'MacOS' if sys.platform == 'darwin' else 'Linux-x86-64'


class OfxPlugin(ctypes.Structure):
    _fields_ = [('pluginApi', ctypes.c_char_p), ('apiVersion', ctypes.c_int), ('pluginIdentifier', ctypes.c_char_p),
                ('pluginVersionMajor', ctypes.c_uint), ('pluginVersionMinor', ctypes.c_uint),
                ('setHost', ctypes.c_void_p), ('mainEntry', ctypes.c_void_p)]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'dist/SigmaFpRaw.ofx.bundle/Contents' / FOLDER / 'SigmaFpRaw.ofx'
    lib = ctypes.CDLL(str(path))
    lib.OfxGetNumberOfPlugins.restype = ctypes.c_int
    lib.OfxGetPlugin.restype = ctypes.POINTER(OfxPlugin)
    lib.OfxGetPlugin.argtypes = [ctypes.c_int]
    count = lib.OfxGetNumberOfPlugins()
    plugin = lib.OfxGetPlugin(0).contents
    print(f'{path}: {count} plug-in, {plugin.pluginIdentifier.decode()} ({plugin.pluginApi.decode()} v{plugin.apiVersion}), '
          f'version {plugin.pluginVersionMajor}.{plugin.pluginVersionMinor}')
    if count != 1 or plugin.pluginIdentifier != b'com.sigmafpmods.raw' or not plugin.mainEntry:
        sys.exit('FAIL')
    print('PASS')


if __name__ == '__main__':
    main()
