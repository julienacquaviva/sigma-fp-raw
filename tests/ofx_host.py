"""Small independent OpenFX 1.4 CPU host for native plug-in integration tests.

This is deliberately not a Resolve emulator. It implements the property,
parameter, image and message suites used by the plug-in, including real image
strides, cropped image bounds and acquired/released image tracking. Calling a
native library can crash Python; run the integration test in a separate process.
"""
from __future__ import annotations

import copy
import ctypes as C
import traceback
from pathlib import Path

import numpy as np

P, S, I, D = C.c_void_p, C.c_char_p, C.c_int, C.c_double
PP = C.POINTER(P)
OK, FAILED, UNKNOWN, BAD_HANDLE, BAD_INDEX, UNSUPPORTED = 0, 1, 3, 9, 10, 5
DEFAULT = 14


class RectD(C.Structure):
    _fields_ = [(n, D) for n in ("x1", "y1", "x2", "y2")]


FetchSuite = C.CFUNCTYPE(P, P, S, I)


class HostStruct(C.Structure):
    _fields_ = [("host", P), ("fetchSuite", FetchSuite)]


SetHost = C.CFUNCTYPE(None, C.POINTER(HostStruct))
MainEntry = C.CFUNCTYPE(I, S, P, P, P)


class Plugin(C.Structure):
    _fields_ = [
        ("pluginApi", S), ("apiVersion", I), ("pluginIdentifier", S),
        ("pluginVersionMajor", C.c_uint), ("pluginVersionMinor", C.c_uint),
        ("setHost", SetHost), ("mainEntry", MainEntry),
    ]


class ImageBuffer:
    """RGBA floats with guard bands around allocation AND each padded row."""

    def __init__(self, bounds, negative_stride=False):
        self.bounds = tuple(bounds)
        x1, y1, x2, y2 = bounds
        self.width, self.height = x2 - x1, y2 - y1
        self.guard, self.padding = 128, 64
        self.pitch = self.width * 16 + self.padding
        self.raw = (C.c_ubyte * (2 * self.guard + self.pitch * self.height))()
        C.memset(C.addressof(self.raw), 0xCD, C.sizeof(self.raw))
        self.row_bytes = -self.pitch if negative_stride else self.pitch
        offset = self.guard + (self.pitch * (self.height - 1) if negative_stride else 0)
        self.pointer = C.addressof(self.raw) + offset
        self.pixels = np.ndarray(
            (self.height, self.width, 4), dtype=np.float32, buffer=self.raw,
            offset=offset, strides=(self.row_bytes, 16, 4),
        )
        self.pixels[:] = -777.0

    def assert_guards(self):
        raw = np.frombuffer(self.raw, dtype=np.uint8)
        assert np.all(raw[:self.guard] == 0xCD), "write before image allocation"
        assert np.all(raw[-self.guard:] == 0xCD), "write after image allocation"
        for y in range(self.height):
            p = self.guard + y * self.pitch + self.width * 16
            assert np.all(raw[p:p + self.padding] == 0xCD), f"write into row {y} padding"


class OFXHost:
    def __init__(self, plugin_path: Path, canvas=(320, 180), message_version=2):
        self.canvas = tuple(canvas)
        self.message_version = message_version
        self.objects = {}
        self.next_handle = 0x10000
        self.callbacks = []
        self.strings = []
        self.callback_errors = []
        self.messages = []
        self.acquired = {}
        self.source_image_requests = 0
        self.suites = {}
        self.host_props = self.props({
            "OfxPropName": ["org.codex.IndependentOFXTestHost"],
            "OfxPropLabel": ["Independent OFX Test Host"],
            "OfxPropVersion": [1, 4],
            "OfxImageEffectHostPropIsBackground": [1],
            "OfxImageEffectPropSupportsTiles": [1],
            "OfxImageEffectPropSupportsMultiResolution": [1],
            "OfxImageEffectPropSupportedComponents": ["OfxImageComponentRGBA"],
            "OfxImageEffectPropSupportedPixelDepths": ["OfxBitDepthFloat"],
        })
        self._build_suites()
        self.fetch_callback = FetchSuite(self._fetch_suite)
        self.struct = HostStruct(self.host_props, self.fetch_callback)
        self.library = C.CDLL(str(Path(plugin_path).resolve()))
        self.library.OfxGetNumberOfPlugins.restype = I
        assert self.library.OfxGetNumberOfPlugins() == 1
        self.library.OfxGetPlugin.argtypes = [I]
        self.library.OfxGetPlugin.restype = C.POINTER(Plugin)
        self.plugin = self.library.OfxGetPlugin(0).contents
        assert self.plugin.pluginApi == b"OfxImageEffectPluginAPI"
        self.plugin.setHost(C.byref(self.struct))
        self.expect("OfxActionLoad", None)
        self.descriptor = self._effect()
        self.expect("OfxActionDescribe", self.descriptor)
        args = self.props({"OfxImageEffectPropContext": ["OfxImageEffectContextFilter"]})
        self.expect("OfxImageEffectActionDescribeInContext", self.descriptor, args)
        self.instance = self._instantiate(self.descriptor)
        self.expect("OfxActionCreateInstance", self.instance)

    def obj(self, value):
        return self.objects[int(value)]

    def add(self, value):
        self.next_handle += 16
        self.objects[self.next_handle] = value
        return self.next_handle

    def props(self, values=None):
        return self.add({"kind": "props", "values": copy.deepcopy(values or {})})

    def _effect(self):
        properties = self.props({
            "OfxImageEffectPropContext": ["OfxImageEffectContextFilter"],
            "OfxImageEffectPropProjectSize": list(map(float, self.canvas)),
            "OfxImageEffectPropProjectExtent": list(map(float, self.canvas)),
            "OfxImageEffectPropProjectOffset": [0.0, 0.0],
            "OfxImageEffectPropProjectPixelAspectRatio": [1.0],
            "OfxImageEffectPropFrameRate": [24.0],
            "OfxImageEffectPropFrameRange": [0.0, 100.0],
            "OfxImageEffectPropRenderScale": [1.0, 1.0],
        })
        params = self.add({"kind": "paramset", "props": properties, "params": {}})
        return self.add({"kind": "effect", "props": properties, "params": params, "clips": {}})

    def _instantiate(self, descriptor):
        result = self._effect()
        desc, inst = self.obj(descriptor), self.obj(result)
        for name, handle in self.obj(desc["params"])["params"].items():
            param = copy.deepcopy(self.obj(handle))
            param["props"] = self.props(self.obj(param["props"])["values"])
            param["value"] = self.obj(param["props"])["values"].get("OfxParamPropDefault", [0])[0]
            self.obj(inst["params"])["params"][name] = self.add(param)
        for name, handle in desc["clips"].items():
            clip = copy.deepcopy(self.obj(handle))
            clip["props"] = self.props(self.obj(clip["props"])["values"])
            p = self.obj(clip["props"])["values"]
            p.update({
                "OfxImageEffectPropComponents": ["OfxImageComponentRGBA"],
                "OfxImageEffectPropPixelDepth": ["OfxBitDepthFloat"],
                "OfxImageEffectPropUnmappedComponents": ["OfxImageComponentRGBA"],
                "OfxImageEffectPropUnmappedPixelDepth": ["OfxBitDepthFloat"],
                "OfxImageEffectPropPixelAspectRatio": [1.0],
                "OfxImageEffectPropFrameRange": [0.0, 999999.0],   # an untrimmed clip; see set_clip_range
                "OfxImageClipPropFrameRange": [0.0, 999999.0],
                "OfxImageClipPropConnected": [1],
            })
            inst["clips"][name] = self.add(clip)
        return result

    def _callback(self, name, return_type, args, func):
        def wrapped(*values):
            try:
                return func(*values)
            except Exception:
                self.callback_errors.append(name + "\n" + traceback.format_exc())
                return FAILED
        result = C.CFUNCTYPE(return_type, *args)(wrapped)
        self.callbacks.append(result)
        return C.cast(result, P).value

    def _suite(self, name, version, functions):
        fields = [(f[0], P) for f in functions]
        suite_type = type(name + str(version), (C.Structure,), {"_fields_": fields})
        values = [self._callback(n, r, a, f) for n, r, a, f in functions]
        self.suites[(name, version)] = suite_type(*values)

    def _fetch_suite(self, host, name, version):
        if name == b"OfxMessageSuite" and version > self.message_version:
            return None
        suite = self.suites.get((name.decode(), version))
        return C.addressof(suite) if suite is not None else None

    def _string(self, value):
        buffer = C.create_string_buffer(value.encode("utf-8") if isinstance(value, str) else value)
        self.strings.append(buffer)
        return C.addressof(buffer)

    def _build_suites(self):
        functions = []
        types = [("Pointer", P), ("String", S), ("Double", D), ("Int", I)]
        for multiple in (False, True):
            for label, typ in types:
                def setter(handle, key, index, value, multiple=multiple, typ=typ):
                    vals = self.obj(handle)["values"]
                    key = key.decode()
                    incoming = list(value[:index]) if multiple else [value]
                    if typ is S:
                        incoming = [v.decode("utf-8") if v else "" for v in incoming]
                    if multiple:
                        vals[key] = incoming
                    else:
                        array = vals.setdefault(key, [])
                        array.extend([None] * max(0, index + 1 - len(array)))
                        array[index] = incoming[0]
                    return OK
                functions.append(("propSet" + label + ("N" if multiple else ""), I,
                                  [P, S, I, C.POINTER(typ) if multiple else typ], setter))
        for multiple in (False, True):
            for label, typ in types:
                outtype = P if typ is S else typ
                def getter(handle, key, index, out, multiple=multiple, typ=typ):
                    vals = self.obj(handle)["values"]
                    key = key.decode()
                    if key not in vals:
                        return UNKNOWN
                    source = vals[key]
                    indexes = range(index) if multiple else [index]
                    for n, k in enumerate(indexes):
                        if k >= len(source):
                            return BAD_INDEX
                        value = source[k]
                        out[n] = self._string(value) if typ is S else value
                    return OK
                functions.append(("propGet" + label + ("N" if multiple else ""), I,
                                  [P, S, I, C.POINTER(outtype)], getter))
        def reset(handle, key):
            self.obj(handle)["values"].pop(key.decode(), None)
            return OK
        def dimension(handle, key, out):
            vals = self.obj(handle)["values"]
            if key.decode() not in vals:
                return UNKNOWN
            out[0] = len(vals[key.decode()])
            return OK
        functions += [("propReset", I, [P, S], reset),
                      ("propGetDimension", I, [P, S, C.POINTER(I)], dimension)]
        self._suite("OfxPropertySuite", 1, functions)

        def get_props(handle, out):
            out[0] = self.obj(handle)["props"]
            return OK
        def get_params(handle, out):
            out[0] = self.obj(handle)["params"]
            return OK
        def clip_define(handle, name, out):
            p = self.props({"OfxPropName": [name.decode()]})
            clip = self.add({"kind": "clip", "name": name.decode(), "props": p})
            self.obj(handle)["clips"][name.decode()] = clip
            out[0] = p
            return OK
        def clip_handle(handle, name, out, props):
            value = self.obj(handle)["clips"].get(name.decode())
            if value is None:
                return UNKNOWN
            out[0] = value
            if props:
                props[0] = self.obj(value)["props"]
            return OK
        def clip_image(handle, time, region, out):
            clip = self.obj(handle)
            if clip["name"] == "Source":
                self.source_image_requests += 1
            image = clip.get("image")
            if image is None:
                return FAILED
            out[0] = image
            self.acquired[image] = self.acquired.get(image, 0) + 1
            return OK
        def release_image(handle):
            assert self.acquired.get(handle, 0) > 0, "image released without acquisition"
            self.acquired[handle] -= 1
            return OK
        def rod(handle, time, out):
            out[0] = RectD(0, 0, *self.canvas)
            return OK
        def unsupported(*args):
            return UNSUPPORTED
        self._suite("OfxImageEffectSuite", 1, [
            ("getPropertySet", I, [P, PP], get_props),
            ("getParamSet", I, [P, PP], get_params),
            ("clipDefine", I, [P, S, PP], clip_define),
            ("clipGetHandle", I, [P, S, PP, PP], clip_handle),
            ("clipGetPropertySet", I, [P, PP], get_props),
            ("clipGetImage", I, [P, D, C.POINTER(RectD), PP], clip_image),
            ("clipReleaseImage", I, [P], release_image),
            ("clipGetRegionOfDefinition", I, [P, D, C.POINTER(RectD)], rod),
            ("abort", I, [P], lambda handle: 0),
            ("imageMemoryAlloc", I, [P, C.c_size_t, PP], unsupported),
            ("imageMemoryFree", I, [P], unsupported),
            ("imageMemoryLock", I, [P, PP], unsupported),
            ("imageMemoryUnlock", I, [P], unsupported),
        ])

        def param_define(handle, typ, name, out):
            p = self.props({"OfxPropName": [name.decode()], "OfxParamPropType": [typ.decode()]})
            param = self.add({"kind": "param", "type": typ.decode(), "name": name.decode(), "props": p})
            self.obj(handle)["params"][name.decode()] = param
            if out:
                out[0] = p
            return OK
        def param_handle(handle, name, out, props):
            param = self.obj(handle)["params"].get(name.decode())
            if param is None:
                return UNKNOWN
            out[0] = param
            if props:
                props[0] = self.obj(param)["props"]
            return OK
        def param_value(handle, out):
            param = self.obj(handle)
            typ, value = param["type"], param["value"]
            if typ == "OfxParamTypeString":
                C.cast(out, PP)[0] = self._string(value)
            elif typ == "OfxParamTypeDouble":
                C.cast(out, C.POINTER(D))[0] = float(value)
            elif typ in ("OfxParamTypeInteger", "OfxParamTypeBoolean", "OfxParamTypeChoice"):
                C.cast(out, C.POINTER(I))[0] = int(value)
            else:
                return UNSUPPORTED
            return OK
        def param_set(handle, bits):
            # Variadic: on Win64 the one value arrives in the second integer register
            # (a double is duplicated there), so its type follows from the parameter.
            param = self.obj(handle)
            typ = param["type"]
            if typ == "OfxParamTypeString":
                param["value"] = C.string_at(bits).decode("utf-8", errors="replace") if bits else ""
            elif typ == "OfxParamTypeDouble":
                param["value"] = C.c_double.from_buffer_copy(C.c_uint64(bits)).value
            elif typ in ("OfxParamTypeInteger", "OfxParamTypeBoolean", "OfxParamTypeChoice"):
                param["value"] = C.c_int32(bits & 0xFFFFFFFF).value
            else:
                return UNSUPPORTED
            return OK
        self._suite("OfxParameterSuite", 1, [
            ("paramDefine", I, [P, S, S, PP], param_define),
            ("paramGetHandle", I, [P, S, PP, PP], param_handle),
            ("paramSetGetPropertySet", I, [P, PP], get_props),
            ("paramGetPropertySet", I, [P, PP], get_props),
            # The variadic tail is one output pointer for every supported type.
            ("paramGetValue", I, [P, P], param_value),
            ("paramGetValueAtTime", I, [P, D, P], lambda h, t, out: param_value(h, out)),
            ("paramGetDerivative", I, [P, D, P], unsupported),
            ("paramGetIntegral", I, [P, D, D, P], unsupported),
            ("paramSetValue", I, [P, C.c_uint64], param_set),
            ("paramSetValueAtTime", I, [P, D], unsupported),
            ("paramGetNumKeys", I, [P, C.POINTER(C.c_uint)], unsupported),
            ("paramGetKeyTime", I, [P, C.c_uint, C.POINTER(D)], unsupported),
            ("paramGetKeyIndex", I, [P, D, I, C.POINTER(I)], unsupported),
            ("paramDeleteKey", I, [P, D], unsupported),
            ("paramDeleteAllKeys", I, [P], unsupported),
            ("paramCopy", I, [P, P, D, P], unsupported),
            ("paramEditBegin", I, [P, S], lambda h, n: OK),
            ("paramEditEnd", I, [P], lambda h: OK),
        ])
        def message(handle, typ, msgid, fmt, arg):
            # The plug-in emits "%s" with one string argument. Other formats
            # are retained verbatim, without dereferencing unknown varargs.
            text = C.string_at(arg).decode("utf-8", errors="replace") if fmt == b"%s" and arg else fmt.decode()
            self.messages.append({"type": typ.decode(), "id": msgid.decode() if msgid else "", "text": text})
            return OK
        self._suite("OfxMessageSuite", 2, [
            ("message", I, [P, S, S, S, P], message),
            ("setPersistentMessage", I, [P, S, S, S, P], message),
            ("clearPersistentMessage", I, [P], lambda h: OK),
        ])
        self._suite("OfxMessageSuite", 1, [
            ("message", I, [P, S, S, S, P], message),
        ])

    def params(self):
        return self.obj(self.obj(self.instance)["params"])["params"]

    def set(self, **values):
        for name, value in values.items():
            assert name in self.params(), f"undeclared parameter {name}"
            self.obj(self.params()[name])["value"] = value

    def get(self, name):
        return self.obj(self.params()[name])["value"]

    def changed(self, name, time=0.0):
        """Tells the plug-in that the user edited a parameter (OfxActionInstanceChanged)."""
        args = self.props({"OfxPropType": ["OfxTypeParameter"], "OfxPropName": [name],
                           "OfxPropChangeReason": ["OfxChangeUserEdited"], "OfxPropTime": [float(time)],
                           "OfxImageEffectPropRenderScale": [1.0, 1.0]})
        return self.action("OfxActionInstanceChanged", self.instance, args)

    def choices(self, name):
        p = self.obj(self.params()[name])["props"]
        return self.obj(p)["values"].get("OfxParamPropChoiceOption", [])

    def action(self, name, handle=None, in_args=None, out_args=None):
        result = self.plugin.mainEntry(name.encode(), handle, in_args, out_args)
        assert not self.callback_errors, "\n".join(self.callback_errors)
        return result

    def expect(self, name, handle=None, in_args=None, out_args=None):
        result = self.action(name, handle, in_args, out_args)
        assert result == OK, f"{name}: status {result}; messages={self.messages[-3:]}"
        return result

    def set_clip_range(self, first, last):
        """Frame range of the Source clip in effect time (what a host reports for a timeline clip)."""
        clip = self.obj(self.instance)["clips"]["Source"]
        self.obj(self.obj(clip)["props"])["values"]["OfxImageEffectPropFrameRange"] = [float(first), float(last)]

    def render(self, *, time=None, source_frame=0, scale=(1.0, 1.0), bounds=None, window=None,
               negative_stride=False, pixel_aspect=1.0, field="OfxFieldNone"):
        if time is None:                      # one source frame per frame of effect time
            time = float(source_frame) if source_frame is not None else 0.0
        if bounds is None:
            bounds = (0, 0, round(self.canvas[0] * scale[0]), round(self.canvas[1] * scale[1]))
        window = tuple(window or bounds)
        output = ImageBuffer(bounds, negative_stride=negative_stride)
        image = self.props({
            "OfxImagePropData": [output.pointer],
            "OfxImagePropBounds": list(bounds),
            "OfxImagePropRegionOfDefinition": [0, 0, round(self.canvas[0] * scale[0]), round(self.canvas[1] * scale[1])],
            "OfxImagePropRowBytes": [output.row_bytes],
            "OfxImageEffectPropComponents": ["OfxImageComponentRGBA"],
            "OfxImageEffectPropPixelDepth": ["OfxBitDepthFloat"],
            "OfxImageEffectPropPreMultiplication": ["OfxImageOpaque"],
            "OfxImagePropPixelAspectRatio": [float(pixel_aspect)],
            "OfxImageEffectPropRenderScale": list(scale),
            "OfxImagePropField": [field],
        })
        clip = self.obj(self.instance)["clips"]["Output"]
        self.obj(clip)["image"] = image
        arg_values = {
            "OfxPropTime": [float(time)],
            "OfxImageEffectPropRenderScale": list(scale),
            "OfxImageEffectPropRenderWindow": list(window),
            "OfxImageEffectPropFieldToRender": ["OfxFieldNone"],
            "OfxImageEffectPropSequentialRenderStatus": [0],
            "OfxImageEffectPropInteractiveRenderStatus": [0],
        }
        if source_frame is not None:
            arg_values["OfxImageEffectPropSrcFrame"] = [int(source_frame)]
        args = self.props(arg_values)
        result = self.action("OfxImageEffectActionRender", self.instance, args)
        output.assert_guards()
        assert all(n == 0 for n in self.acquired.values()), "plug-in failed to release an acquired image"
        return result, output

    def close(self):
        if getattr(self, "instance", None):
            self.expect("OfxActionDestroyInstance", self.instance)
            self.instance = None
        result = self.action("OfxActionUnload")
        assert result in (OK, DEFAULT), result
