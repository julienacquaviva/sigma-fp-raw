// Minimal CUDA driver API, loaded at run time from the NVIDIA driver (nvcuda.dll, libcuda.so.1;
// no toolkit needed). Without it cuda().ok is false and frames are developed on the CPU.
#pragma once
#include <cstddef>
#include <string>

namespace sfp {

typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef unsigned long long CUdeviceptr;

struct Cuda {
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDeviceGet)(CUdevice*, int);
    CUresult (*cuDeviceGetName)(char*, int, CUdevice);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
    CUresult (*cuCtxGetCurrent)(CUcontext*);
    CUresult (*cuCtxPushCurrent)(CUcontext);
    CUresult (*cuCtxPopCurrent)(CUcontext*);
    CUresult (*cuCtxSynchronize)();
    CUresult (*cuModuleLoadData)(CUmodule*, const void*);
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                               unsigned, CUstream, void**, void**);
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemAllocHost)(void**, size_t);
    CUresult (*cuMemFreeHost)(void*);
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream);
    CUresult (*cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuStreamCreate)(CUstream*, unsigned);
    CUresult (*cuGetErrorString)(CUresult, const char**);
    bool ok = false;
    std::string error;
};

// Loads the driver once; returns the shared table (ok == false with error when absent).
Cuda& cuda();
std::string cuda_error(CUresult r);

}  // namespace sfp
