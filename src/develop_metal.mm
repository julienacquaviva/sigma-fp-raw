// Development on the Mac's GPU with Metal: the kernels of kernels.cu, compiled from their source
// text when the plug-in first renders, run on the system's default GPU. Same picture as the
// processor path; host images in ordinary memory go in and out (on Apple Silicon the GPU shares
// that memory, so this costs one copy each way).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "develop_metal.h"
#include "kernels_msl.h"
#include "platform.h"

namespace sfp {
namespace {

struct Arg {
    id<MTLBuffer> buffer = nil;
    const void* bytes = nullptr;
    size_t size = 0;
    Arg(id<MTLBuffer> b) : buffer(b) {}
    Arg(const void* p, size_t n) : bytes(p), size(n) {}
};
template <class T> Arg value(const T& v) { return Arg(&v, sizeof v); }

struct State {
    std::mutex m;
    bool tried = false, ok = false;
    std::string error, name;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> hot = nil, prep = nil, vh = nil, lpf = nil, green = nil, pq = nil, rbDiag = nil, rbG = nil, half = nil,
                                tensor = nil, blur3 = nil, dezig = nil, dev = nil, unpack = nil;
    id<MTLBuffer> raw = nil, raw2 = nil, cfa = nil, vhb = nil, lpfb = nil, pqb = nil, R = nil, G = nil, B = nil;
    id<MTLBuffer> T[6] = {nil, nil, nil, nil, nil, nil};
    id<MTLBuffer> planes[3] = {nil, nil, nil};
    id<MTLBuffer> out = nil, src = nil;
    size_t cap = 0, outCap = 0, srcCap = 0;
    // Demosaic reuse key.
    bool valid = false;
    uint64_t frameId = 0;
    int quality = -1;
    float lastDezig = -1;
    PrepParams last{};
};

State& state() {
    static State* s = new State;   // kept for the life of the process
    return *s;
}

std::string text(NSError* err) {
    if (!err) return "unknown error";
    NSString* d = err.localizedDescription;
    return d ? std::string(d.UTF8String) : "unknown error";
}

// Creates the device, compiles the kernels and makes one pipeline per kernel. Once.
bool init(State& x) {
    if (x.tried) return x.ok;
    x.tried = true;
    if (os::env("SFP_METAL") == "0") { x.error = "switched off (SFP_METAL=0)"; return false; }
    @autoreleasepool {
        // The default device needs Core Graphics linked in (a command-line tool has none by
        // itself); should it still be missing, any device of the machine will do.
        x.device = MTLCreateSystemDefaultDevice();
        if (!x.device) {
            NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
            if (all.count) x.device = all[0];
        }
        if (!x.device) { x.error = "no Metal device"; return false; }
        x.name = x.device.name ? std::string(x.device.name.UTF8String) : "GPU";
        x.queue = [x.device newCommandQueue];
        if (!x.queue) { x.error = "no Metal command queue"; return false; }
        MTLCompileOptions* options = [MTLCompileOptions new];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;   // the same arithmetic as the processor path
#pragma clang diagnostic pop
        NSError* err = nil;
        id<MTLLibrary> lib = [x.device newLibraryWithSource:[NSString stringWithUTF8String:kKernelsMsl] options:options error:&err];
        if (!lib) { x.error = "kernels do not compile: " + text(err); return false; }
        struct { id<MTLComputePipelineState> __strong* slot; const char* fn; } table[] = {
            {&x.hot, "k_hot"}, {&x.prep, "k_prep"}, {&x.vh, "k_vh"}, {&x.lpf, "k_lpf"}, {&x.green, "k_green"}, {&x.pq, "k_pq"},
            {&x.rbDiag, "k_rb_diag"}, {&x.rbG, "k_rb_g"}, {&x.half, "k_half"}, {&x.tensor, "k_tensor"}, {&x.blur3, "k_blur3"},
            {&x.dezig, "k_dezig"}, {&x.dev, "k_develop"}, {&x.unpack, "k_unpack"}};
        for (auto& t : table) {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:t.fn]];
            if (!fn) { x.error = std::string("kernel ") + t.fn + " not found"; return false; }
            err = nil;
            *t.slot = [x.device newComputePipelineStateWithFunction:fn error:&err];
            if (!*t.slot) { x.error = std::string("kernel ") + t.fn + ": " + text(err); return false; }
        }
    }
    x.ok = true;
    return true;
}

bool make(State& x, id<MTLBuffer> __strong& b, size_t bytes, std::string& e) {
    b = [x.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) { e = "Not enough GPU memory"; return false; }
    return true;
}

// One kernel over w x h positions; the arguments in the kernel's order.
void launch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps, int w, int h, std::initializer_list<Arg> args) {
    [enc setComputePipelineState:ps];
    NSUInteger i = 0;
    for (const Arg& a : args) {
        if (a.buffer) [enc setBuffer:a.buffer offset:0 atIndex:i];
        else [enc setBytes:a.bytes length:a.size atIndex:i];
        ++i;
    }
    const NSUInteger tw = ps.threadExecutionWidth, th = std::max<NSUInteger>(1, ps.maxTotalThreadsPerThreadgroup / tw);
    [enc dispatchThreadgroups:MTLSizeMake((w + tw - 1) / tw, (h + th - 1) / th, 1) threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

}  // namespace

bool metal_available() {
    State& x = state();
    std::lock_guard<std::mutex> lock(x.m);
    return init(x);
}

std::string metal_status() {
    State& x = state();
    std::lock_guard<std::mutex> lock(x.m);
    init(x);
    return x.ok ? x.name : x.error;
}

bool develop_metal(const Frame& f, int quality, const PrepParams& pp, float dz, const DevelopParams& dpIn, void* out,
                   std::string& e, DevelopTiming* timing, const void* source, int sourceRowBytes) {
    State& x = state();
    std::lock_guard<std::mutex> lock(x.m);
    if (!init(x)) { e = "Metal: " + x.error; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    @autoreleasepool {
        const int W = source ? dpIn.W : pp.W, H = source ? dpIn.H : pp.H;
        const size_t n = static_cast<size_t>(W) * H;
        if (x.cap < n) {
            x.cap = 0;
            x.valid = false;
            if (!make(x, x.raw, n * 2, e) || !make(x, x.raw2, n * 2, e)) return false;
            id<MTLBuffer> __strong* planes[] = {&x.cfa, &x.vhb, &x.lpfb, &x.pqb, &x.R, &x.G, &x.B, &x.T[0], &x.T[1], &x.T[2], &x.T[3], &x.T[4], &x.T[5]};
            for (id<MTLBuffer> __strong* b : planes)
                if (!make(x, *b, n * 4, e)) return false;
            x.cap = n;
        }
        // The kernels write a packed image; its rows go to the host image afterwards.
        DevelopParams dp = dpIn;
        const size_t packed = static_cast<size_t>(dp.outW) * 16, outBytes = packed * dp.outH;
        dp.rowBytes = static_cast<int>(packed);
        if (x.outCap < outBytes) {
            x.outCap = 0;
            if (!make(x, x.out, outBytes, e)) return false;
            x.outCap = outBytes;
        }
        id<MTLCommandBuffer> cb = [x.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!cb || !enc) { e = "Metal: no command buffer"; return false; }
        const bool reuse = !source && x.valid && x.frameId == f.id && x.quality == quality && x.lastDezig == dz && !std::memcmp(&x.last, &pp, sizeof pp);
        if (source) {
            x.valid = false;
            const size_t srcPacked = static_cast<size_t>(W) * 16, bytes = srcPacked * H;
            if (x.srcCap < bytes) {
                x.srcCap = 0;
                if (!make(x, x.src, bytes, e)) { [enc endEncoding]; return false; }
                x.srcCap = bytes;
            }
            char* to = static_cast<char*>(x.src.contents);
            for (int y = 0; y < H; ++y)
                std::memcpy(to + y * srcPacked, static_cast<const char*>(source) + static_cast<long long>(y) * sourceRowBytes, srcPacked);
            const int rowBytes = static_cast<int>(srcPacked);
            launch(enc, x.unpack, W, H, {x.src, x.R, x.G, x.B, value(W), value(H), value(rowBytes)});
            x.planes[0] = x.R; x.planes[1] = x.G; x.planes[2] = x.B;
        } else if (!reuse) {
            x.valid = false;
            std::memcpy(x.raw.contents, f.raw.data(), n * 2);
            id<MTLBuffer> src = x.raw;
            if (pp.hotPixels) {
                launch(enc, x.hot, W, H, {x.raw, x.raw2, value(pp)});
                src = x.raw2;
            }
            launch(enc, x.prep, W, H, {src, x.cfa, value(pp)});
            if (quality == 1) {
                launch(enc, x.half, W / 2, H / 2, {x.cfa, x.R, x.G, x.B, value(pp)});
            } else {
                launch(enc, x.vh, W, H, {x.cfa, x.vhb, value(pp.W), value(pp.H)});
                launch(enc, x.lpf, W, H, {x.cfa, x.lpfb, value(pp)});
                launch(enc, x.green, W, H, {x.cfa, x.vhb, x.lpfb, x.G, value(pp)});
                launch(enc, x.pq, W, H, {x.cfa, x.pqb, value(pp)});
                launch(enc, x.rbDiag, W, H, {x.cfa, x.G, x.pqb, x.R, x.B, value(pp)});
                launch(enc, x.rbG, W, H, {x.G, x.vhb, x.R, x.B, value(pp)});
            }
            x.planes[0] = x.R; x.planes[1] = x.G; x.planes[2] = x.B;
            if (dz > 0.f) {
                // Demosaic scratch (cfa, vh, lpf) is free now: de-zigzag into it.
                const DezigParams zp{W, H, dz, 12.0f, 0.08f, 0.45f, 2e-5f};
                const int d0 = 0, d1 = 1;
                launch(enc, x.tensor, W, H, {x.R, x.G, x.B, x.T[0], x.T[1], x.T[2], value(pp.W), value(pp.H)});
                launch(enc, x.blur3, W, H, {x.T[0], x.T[1], x.T[2], x.T[3], x.T[4], x.T[5], value(pp.W), value(pp.H), value(d0)});
                launch(enc, x.blur3, W, H, {x.T[3], x.T[4], x.T[5], x.T[0], x.T[1], x.T[2], value(pp.W), value(pp.H), value(d1)});
                launch(enc, x.dezig, W, H, {x.R, x.G, x.B, x.T[0], x.T[1], x.T[2], x.cfa, x.vhb, x.lpfb, value(zp)});
                x.planes[0] = x.cfa; x.planes[1] = x.vhb; x.planes[2] = x.lpfb;
            }
        }
        launch(enc, x.dev, dp.outW, dp.outH, {x.planes[0], x.planes[1], x.planes[2], x.out, value(dp)});
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            x.valid = false;
            e = "Metal: " + text(cb.error);
            return false;
        }
        if (!source && !reuse) {
            x.lastDezig = dz;
            x.valid = true;
            x.frameId = f.id;
            x.quality = quality;
            x.last = pp;
        }
        const char* from = static_cast<const char*>(x.out.contents);
        for (int y = 0; y < dp.outH; ++y)
            std::memcpy(static_cast<char*>(out) + static_cast<long long>(y) * dpIn.rowBytes, from + y * packed, packed);
        if (timing) timing->demosaicReused = reuse;
    }
    if (timing) timing->gpuMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace sfp
