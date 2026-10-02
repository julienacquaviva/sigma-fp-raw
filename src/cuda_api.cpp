#include "cuda_api.h"

#include <mutex>
#include <type_traits>

#include "platform.h"

namespace sfp {

Cuda& cuda() {
    static Cuda c;
    static std::once_flag once;
    std::call_once(once, [] {
#if defined(_WIN32)
        void* m = os::load_library("nvcuda.dll");
#elif defined(__APPLE__)
        void* m = nullptr;   // no CUDA on macOS
#else
        void* m = os::load_library("libcuda.so.1");
        if (!m) m = os::load_library("libcuda.so");
#endif
        if (!m) { c.error = "NVIDIA CUDA driver not found"; return; }
        bool missing = false;
        auto get = [&](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(os::library_symbol(m, name));
            if (!fn) { missing = true; c.error = std::string("CUDA driver lacks ") + name; }
        };
        get(c.cuInit, "cuInit");
        get(c.cuDeviceGet, "cuDeviceGet");
        get(c.cuDeviceGetName, "cuDeviceGetName");
        get(c.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
        get(c.cuCtxGetCurrent, "cuCtxGetCurrent");
        get(c.cuCtxPushCurrent, "cuCtxPushCurrent_v2");
        get(c.cuCtxPopCurrent, "cuCtxPopCurrent_v2");
        get(c.cuCtxSynchronize, "cuCtxSynchronize");
        get(c.cuModuleLoadData, "cuModuleLoadData");
        get(c.cuModuleGetFunction, "cuModuleGetFunction");
        get(c.cuLaunchKernel, "cuLaunchKernel");
        get(c.cuMemAlloc, "cuMemAlloc_v2");
        get(c.cuMemFree, "cuMemFree_v2");
        get(c.cuMemAllocHost, "cuMemAllocHost_v2");
        get(c.cuMemFreeHost, "cuMemFreeHost");
        get(c.cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
        get(c.cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
        get(c.cuStreamSynchronize, "cuStreamSynchronize");
        get(c.cuStreamCreate, "cuStreamCreate");
        get(c.cuGetErrorString, "cuGetErrorString");
        if (missing) return;
        CUresult r = c.cuInit(0);
        if (r) { c.error = "cuInit failed: " + cuda_error(r); return; }
        c.ok = true;
    });
    return c;
}

std::string cuda_error(CUresult r) {
    const char* s = nullptr;
    auto& c = cuda();
    if (c.cuGetErrorString && c.cuGetErrorString(r, &s) == 0 && s) return s;
    return "CUDA error " + std::to_string(r);
}

}  // namespace sfp
