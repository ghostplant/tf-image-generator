#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#include <execinfo.h>
#include <sys/time.h>
#include <cuda.h>

#if CUDA_VERSION != 10000
#error "Profiler is only designed for CUDA 10.0"
#endif

static void ktrace(int count) {
    void *array[count];
    size_t size, i;
    char **strings;
    size = backtrace(array, sizeof(array) / sizeof(*array));
    strings = backtrace_symbols(array, size);
    if (NULL == strings)
        perror("backtrace_synbols"), exit(0);
    fprintf (stdout, " - Obtained %zd stack frames.\n", size);
    for (i = 0; i < size; i++)
        fprintf(stdout, "##### %s\n", strings[i]);
    free(strings);
}

#define LOG_API_CALL    // (fprintf(stdout, "<<libcuda.so>> [%lu] call ~%s\n", pthread_self(), __PRETTY_FUNCTION__), fflush(stdout))
#define SYS_CALL(func)    ((hd || (hd = dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.1", RTLD_NOW|RTLD_LOCAL))), assert(hd != NULL), ((CUresult(*)(...))dlsym(hd, func)))
#define _P(func)          LOG_API_CALL; return SYS_CALL(func)
#define _P_LOCAL()        _P(__FUNCTION__)

static void *hd = NULL;

extern "C" {


CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
  LOG_API_CALL;

  static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
  static CUstream hStreams[32];
  int currDev = -1;
  assert(CUDA_SUCCESS == cuCtxGetDevice(&currDev) && currDev >= 0);
  pthread_mutex_lock(&lock);
  if (!hStreams[currDev])
    assert(CUDA_SUCCESS == SYS_CALL(__FUNCTION__)(&hStreams[currDev], Flags)), printf(">> Using single stream for device %d;\n", currDev);
  *phStream = hStreams[currDev];
  // hStreams[currDev] = NULL; // disable single stream
  pthread_mutex_unlock(&lock);

  return CUDA_SUCCESS;
}

CUresult cuStreamDestroy_v2(CUstream hStream) {
  _P_LOCAL()(hStream);
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
  assert(hStream != NULL);
  _P_LOCAL()(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}


CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) { return cuStreamCreate(phStream, flags); }
CUresult cuGetErrorString(CUresult error, const char **pStr) { _P_LOCAL()(error, pStr); }
CUresult cuInit(unsigned int Flags) { _P_LOCAL()(Flags); }
CUresult cuGetErrorName(CUresult error, const char **pStr) { _P_LOCAL()(error, pStr); }
CUresult cuDriverGetVersion(int *driverVersion) { _P_LOCAL()(driverVersion); }
CUresult cuDeviceGet(CUdevice *device, int ordinal) { _P_LOCAL()(device, ordinal); }
CUresult cuDeviceGetCount(int *count) { _P_LOCAL()(count); }
CUresult cuDeviceGetName(char *name, int len, CUdevice dev) { _P_LOCAL()(name, len, dev); }
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) { _P_LOCAL()(uuid, dev); }
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) { _P_LOCAL()(bytes, dev); }
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) { _P_LOCAL()(pi, attrib, dev); }
CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) { _P_LOCAL()(prop, dev); }
CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) { _P_LOCAL()(major, minor, dev); }
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) { _P_LOCAL()(pctx, dev); }
CUresult cuDevicePrimaryCtxRelease(CUdevice dev) { _P_LOCAL()(dev); }
CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) { _P_LOCAL()(dev, flags); }
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) { _P_LOCAL()(dev, flags, active); }
CUresult cuDevicePrimaryCtxReset(CUdevice dev) { _P_LOCAL()(dev); }
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) { _P_LOCAL()(pctx, flags, dev); }
CUresult cuCtxDestroy_v2(CUcontext ctx) { _P_LOCAL()(ctx); }
CUresult cuCtxPushCurrent_v2(CUcontext ctx) { _P_LOCAL()(ctx); }
CUresult cuCtxPopCurrent_v2(CUcontext *pctx) { _P_LOCAL()(pctx); }
CUresult cuCtxSetCurrent(CUcontext ctx) { _P_LOCAL()(ctx); }
CUresult cuCtxGetCurrent(CUcontext *pctx) { _P_LOCAL()(pctx); }
CUresult cuCtxGetDevice(CUdevice *device) { _P_LOCAL()(device); }
CUresult cuCtxGetFlags(unsigned int *flags) { _P_LOCAL()(flags); }
CUresult cuCtxSynchronize(void) { _P_LOCAL()(); }
CUresult cuCtxSetLimit(CUlimit limit, size_t value) { _P_LOCAL()(limit, value); }
CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) { _P_LOCAL()(pvalue, limit); }
CUresult cuCtxGetCacheConfig(CUfunc_cache *pconfig) { _P_LOCAL()(pconfig); }
CUresult cuCtxSetCacheConfig(CUfunc_cache config) { _P_LOCAL()(config); }
CUresult cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) { _P_LOCAL()(pConfig); }
CUresult cuCtxSetSharedMemConfig(CUsharedconfig config) { _P_LOCAL()(config); }
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) { _P_LOCAL()(ctx, version); }
CUresult cuCtxGetStreamPriorityRange(int *leastPriority, int *greatestPriority) { _P_LOCAL()(leastPriority, greatestPriority); }
CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags) { _P_LOCAL()(pctx, flags); }
CUresult cuCtxDetach(CUcontext ctx) { _P_LOCAL()(ctx); }
CUresult cuModuleLoad(CUmodule *module, const char *fname) { _P_LOCAL()(module, fname); }
CUresult cuModuleLoadData(CUmodule *module, const void *image) { _P_LOCAL()(module, image); }
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues) { _P_LOCAL()(module, image, numOptions, options, optionValues); }
CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) { _P_LOCAL()(module, fatCubin); }
CUresult cuModuleUnload(CUmodule hmod) { _P_LOCAL()(hmod); }
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) { _P_LOCAL()(hfunc, hmod, name); }
CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) { _P_LOCAL()(dptr, bytes, hmod, name); }
CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name) { _P_LOCAL()(pTexRef, hmod, name); }
CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod, const char *name) { _P_LOCAL()(pSurfRef, hmod, name); }
CUresult cuLinkCreate_v2(unsigned int numOptions, CUjit_option *options, void **optionValues, CUlinkState *stateOut) { _P_LOCAL()(numOptions, options, optionValues, stateOut); }
CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues) { _P_LOCAL()(state, type, data, size, name, numOptions, options, optionValues); }
CUresult cuLinkAddFile_v2(CUlinkState state, CUjitInputType type, const char *path, unsigned int numOptions, CUjit_option *options, void **optionValues) { _P_LOCAL()(state, type, path, numOptions, options, optionValues); }
CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) { _P_LOCAL()(state, cubinOut, sizeOut); }
CUresult cuLinkDestroy(CUlinkState state) { _P_LOCAL()(state); }
CUresult cuMemGetInfo_v2(size_t *free, size_t *total) { _P_LOCAL()(free, total); }
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) { _P_LOCAL()(dptr, bytesize); }
CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) { _P_LOCAL()(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes); }
CUresult cuMemFree_v2(CUdeviceptr dptr) { _P_LOCAL()(dptr); }
CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) { _P_LOCAL()(pbase, psize, dptr); }
CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) { _P_LOCAL()(pp, bytesize); }
CUresult cuMemFreeHost(void *p) { _P_LOCAL()(p); }
CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) { _P_LOCAL()(pp, bytesize, Flags); }
CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags) { _P_LOCAL()(pdptr, p, Flags); }
CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) { _P_LOCAL()(pFlags, p); }
CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) { _P_LOCAL()(dptr, bytesize, flags); }
CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) { _P_LOCAL()(dev, pciBusId); }
CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) { _P_LOCAL()(pciBusId, len, dev); }
CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) { _P_LOCAL()(pHandle, event); }
CUresult cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle) { _P_LOCAL()(phEvent, handle); }
CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) { _P_LOCAL()(pHandle, dptr); }
CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags) { _P_LOCAL()(pdptr, handle, Flags); }
CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) { _P_LOCAL()(dptr); }
CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags) { _P_LOCAL()(p, bytesize, Flags); }
CUresult cuMemHostUnregister(void *p) { _P_LOCAL()(p); }
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) { _P_LOCAL()(dst, src, ByteCount); }
CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) { _P_LOCAL()(dstDevice, dstContext, srcDevice, srcContext, ByteCount); }
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) { _P_LOCAL()(dstDevice, srcHost, ByteCount); }
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) { _P_LOCAL()(dstHost, srcDevice, ByteCount); }
CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) { _P_LOCAL()(dstDevice, srcDevice, ByteCount); }
CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount) { _P_LOCAL()(dstArray, dstOffset, srcDevice, ByteCount); }
CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount) { _P_LOCAL()(dstDevice, srcArray, srcOffset, ByteCount); }
CUresult cuMemcpyHtoA_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount) { _P_LOCAL()(dstArray, dstOffset, srcHost, ByteCount); }
CUresult cuMemcpyAtoH_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount) { _P_LOCAL()(dstHost, srcArray, srcOffset, ByteCount); }
CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount) { _P_LOCAL()(dstArray, dstOffset, srcArray, srcOffset, ByteCount); }
CUresult cuMemcpy2D_v2(const CUDA_MEMCPY2D *pCopy) { _P_LOCAL()(pCopy); }
CUresult cuMemcpy2DUnaligned_v2(const CUDA_MEMCPY2D *pCopy) { _P_LOCAL()(pCopy); }
CUresult cuMemcpy3D_v2(const CUDA_MEMCPY3D *pCopy) { _P_LOCAL()(pCopy); }
CUresult cuMemcpy3DPeer(const CUDA_MEMCPY3D_PEER *pCopy) { _P_LOCAL()(pCopy); }
CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream) { _P_LOCAL()(dst, src, ByteCount, hStream); }
CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream) { _P_LOCAL()(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream); }
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) { _P_LOCAL()(dstDevice, srcHost, ByteCount, hStream); }
CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) { _P_LOCAL()(dstHost, srcDevice, ByteCount, hStream); }
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) { _P_LOCAL()(dstDevice, srcDevice, ByteCount, hStream); }
CUresult cuMemcpyHtoAAsync_v2(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream) { _P_LOCAL()(dstArray, dstOffset, srcHost, ByteCount, hStream); }
CUresult cuMemcpyAtoHAsync_v2(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream) { _P_LOCAL()(dstHost, srcArray, srcOffset, ByteCount, hStream); }
CUresult cuMemcpy2DAsync_v2(const CUDA_MEMCPY2D *pCopy, CUstream hStream) { _P_LOCAL()(pCopy, hStream); }
CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy, CUstream hStream) { _P_LOCAL()(pCopy, hStream); }
CUresult cuMemcpy3DPeerAsync(const CUDA_MEMCPY3D_PEER *pCopy, CUstream hStream) { _P_LOCAL()(pCopy, hStream); }
CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) { _P_LOCAL()(dstDevice, uc, N); }
CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) { _P_LOCAL()(dstDevice, us, N); }
CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) { _P_LOCAL()(dstDevice, ui, N); }
CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) { _P_LOCAL()(dstDevice, dstPitch, uc, Width, Height); }
CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) { _P_LOCAL()(dstDevice, dstPitch, us, Width, Height); }
CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) { _P_LOCAL()(dstDevice, dstPitch, ui, Width, Height); }
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream) { _P_LOCAL()(dstDevice, uc, N, hStream); }
CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream) { _P_LOCAL()(dstDevice, us, N, hStream); }
CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N, CUstream hStream) { _P_LOCAL()(dstDevice, ui, N, hStream); }
CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height, CUstream hStream) { _P_LOCAL()(dstDevice, dstPitch, uc, Width, Height, hStream); }
CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream) { _P_LOCAL()(dstDevice, dstPitch, us, Width, Height, hStream); }
CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height, CUstream hStream) { _P_LOCAL()(dstDevice, dstPitch, ui, Width, Height, hStream); }
CUresult cuArrayCreate_v2(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) { _P_LOCAL()(pHandle, pAllocateArray); }
CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray) { _P_LOCAL()(pArrayDescriptor, hArray); }
CUresult cuArrayDestroy(CUarray hArray) { _P_LOCAL()(hArray); }
CUresult cuArray3DCreate_v2(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) { _P_LOCAL()(pHandle, pAllocateArray); }
CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) { _P_LOCAL()(pArrayDescriptor, hArray); }
CUresult cuMipmappedArrayCreate(CUmipmappedArray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc, unsigned int numMipmapLevels) { _P_LOCAL()(pHandle, pMipmappedArrayDesc, numMipmapLevels); }
CUresult cuMipmappedArrayGetLevel(CUarray *pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level) { _P_LOCAL()(pLevelArray, hMipmappedArray, level); }
CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) { _P_LOCAL()(hMipmappedArray); }
CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) { _P_LOCAL()(data, attribute, ptr); }
CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count, CUdevice dstDevice, CUstream hStream) { _P_LOCAL()(devPtr, count, dstDevice, hStream); }
CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice, CUdevice device) { _P_LOCAL()(devPtr, count, advice, device); }
CUresult cuMemRangeGetAttribute(void *data, size_t dataSize, CUmem_range_attribute attribute, CUdeviceptr devPtr, size_t count) { _P_LOCAL()(data, dataSize, attribute, devPtr, count); }
CUresult cuMemRangeGetAttributes(void **data, size_t *dataSizes, CUmem_range_attribute *attributes, size_t numAttributes, CUdeviceptr devPtr, size_t count) { _P_LOCAL()(data, dataSizes, attributes, numAttributes, devPtr, count); }
CUresult cuPointerSetAttribute(const void *value, CUpointer_attribute attribute, CUdeviceptr ptr) { _P_LOCAL()(value, attribute, ptr); }
CUresult cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes, void **data, CUdeviceptr ptr) { _P_LOCAL()(numAttributes, attributes, data, ptr); }
CUresult cuStreamGetPriority(CUstream hStream, int *priority) { _P_LOCAL()(hStream, priority); }
CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) { _P_LOCAL()(hStream, flags); }
CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) { _P_LOCAL()(hStream, pctx); }
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) { _P_LOCAL()(hStream, hEvent, Flags); }
CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback, void *userData, unsigned int flags) { _P_LOCAL()(hStream, callback, userData, flags); }
CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr, size_t length, unsigned int flags) { _P_LOCAL()(hStream, dptr, length, flags); }
CUresult cuStreamQuery(CUstream hStream) { _P_LOCAL()(hStream); }
CUresult cuStreamSynchronize(CUstream hStream) { _P_LOCAL()(hStream); }
CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) { _P_LOCAL()(phEvent, Flags); }
CUresult cuEventRecord(CUevent hEvent, CUstream hStream) { _P_LOCAL()(hEvent, hStream); }
CUresult cuEventQuery(CUevent hEvent) { _P_LOCAL()(hEvent); }
CUresult cuEventSynchronize(CUevent hEvent) { _P_LOCAL()(hEvent); }
CUresult cuEventDestroy_v2(CUevent hEvent) { _P_LOCAL()(hEvent); }
CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) { _P_LOCAL()(pMilliseconds, hStart, hEnd); }
CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) { _P_LOCAL()(stream, addr, value, flags); }
CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) { _P_LOCAL()(stream, addr, value, flags); }
CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags) { _P_LOCAL()(stream, addr, value, flags); }
CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr, cuuint64_t value, unsigned int flags) { _P_LOCAL()(stream, addr, value, flags); }
CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags) { _P_LOCAL()(stream, count, paramArray, flags); }
CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc) { _P_LOCAL()(pi, attrib, hfunc); }
CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value) { _P_LOCAL()(hfunc, attrib, value); }
CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) { _P_LOCAL()(hfunc, config); }
CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) { _P_LOCAL()(hfunc, config); }
CUresult cuLaunchCooperativeKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams) { _P_LOCAL()(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams); }
CUresult cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *launchParamsList, unsigned int numDevices, unsigned int flags) { _P_LOCAL()(launchParamsList, numDevices, flags); }
CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) { _P_LOCAL()(hfunc, x, y, z); }
CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) { _P_LOCAL()(hfunc, bytes); }
CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) { _P_LOCAL()(hfunc, numbytes); }
CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value) { _P_LOCAL()(hfunc, offset, value); }
CUresult cuParamSetf(CUfunction hfunc, int offset, float value) { _P_LOCAL()(hfunc, offset, value); }
CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes) { _P_LOCAL()(hfunc, offset, ptr, numbytes); }
CUresult cuLaunch(CUfunction f) { _P_LOCAL()(f); }
CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) { _P_LOCAL()(f, grid_width, grid_height); }
CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height, CUstream hStream) { _P_LOCAL()(f, grid_width, grid_height, hStream); }
CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) { _P_LOCAL()(hfunc, texunit, hTexRef); }
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) { _P_LOCAL()(numBlocks, func, blockSize, dynamicSMemSize); }
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, unsigned int flags) { _P_LOCAL()(numBlocks, func, blockSize, dynamicSMemSize, flags); }
CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit) { _P_LOCAL()(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit); }
CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(int *minGridSize, int *blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, unsigned int flags) { _P_LOCAL()(minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit, flags); }
CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) { _P_LOCAL()(hTexRef, hArray, Flags); }
CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef, CUmipmappedArray hMipmappedArray, unsigned int Flags) { _P_LOCAL()(hTexRef, hMipmappedArray, Flags); }
CUresult cuTexRefSetAddress_v2(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) { _P_LOCAL()(ByteOffset, hTexRef, dptr, bytes); }
CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch) { _P_LOCAL()(hTexRef, desc, dptr, Pitch); }
CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) { _P_LOCAL()(hTexRef, fmt, NumPackedComponents); }
CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) { _P_LOCAL()(hTexRef, dim, am); }
CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) { _P_LOCAL()(hTexRef, fm); }
CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) { _P_LOCAL()(hTexRef, fm); }
CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) { _P_LOCAL()(hTexRef, bias); }
CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp) { _P_LOCAL()(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp); }
CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) { _P_LOCAL()(hTexRef, maxAniso); }
CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) { _P_LOCAL()(hTexRef, pBorderColor); }
CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) { _P_LOCAL()(hTexRef, Flags); }
CUresult cuTexRefGetAddress_v2(CUdeviceptr *pdptr, CUtexref hTexRef) { _P_LOCAL()(pdptr, hTexRef); }
CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) { _P_LOCAL()(phArray, hTexRef); }
CUresult cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray, CUtexref hTexRef) { _P_LOCAL()(phMipmappedArray, hTexRef); }
CUresult cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim) { _P_LOCAL()(pam, hTexRef, dim); }
CUresult cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) { _P_LOCAL()(pfm, hTexRef); }
CUresult cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef) { _P_LOCAL()(pFormat, pNumChannels, hTexRef); }
CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) { _P_LOCAL()(pfm, hTexRef); }
CUresult cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef) { _P_LOCAL()(pbias, hTexRef); }
CUresult cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp, float *pmaxMipmapLevelClamp, CUtexref hTexRef) { _P_LOCAL()(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef); }
CUresult cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef) { _P_LOCAL()(pmaxAniso, hTexRef); }
CUresult cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef) { _P_LOCAL()(pBorderColor, hTexRef); }
CUresult cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) { _P_LOCAL()(pFlags, hTexRef); }
CUresult cuTexRefCreate(CUtexref *pTexRef) { _P_LOCAL()(pTexRef); }
CUresult cuTexRefDestroy(CUtexref hTexRef) { _P_LOCAL()(hTexRef); }
CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray, unsigned int Flags) { _P_LOCAL()(hSurfRef, hArray, Flags); }
CUresult cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) { _P_LOCAL()(phArray, hSurfRef); }
CUresult cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc, const CUDA_TEXTURE_DESC *pTexDesc, const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) { _P_LOCAL()(pTexObject, pResDesc, pTexDesc, pResViewDesc); }
CUresult cuTexObjectDestroy(CUtexObject texObject) { _P_LOCAL()(texObject); }
CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUtexObject texObject) { _P_LOCAL()(pResDesc, texObject); }
CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc, CUtexObject texObject) { _P_LOCAL()(pTexDesc, texObject); }
CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc, CUtexObject texObject) { _P_LOCAL()(pResViewDesc, texObject); }
CUresult cuSurfObjectCreate(CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc) { _P_LOCAL()(pSurfObject, pResDesc); }
CUresult cuSurfObjectDestroy(CUsurfObject surfObject) { _P_LOCAL()(surfObject); }
CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc, CUsurfObject surfObject) { _P_LOCAL()(pResDesc, surfObject); }
CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) { _P_LOCAL()(canAccessPeer, dev, peerDev); }
CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) { _P_LOCAL()(peerContext, Flags); }
CUresult cuCtxDisablePeerAccess(CUcontext peerContext) { _P_LOCAL()(peerContext); }
CUresult cuDeviceGetP2PAttribute(int* value, CUdevice_P2PAttribute attrib, CUdevice srcDevice, CUdevice dstDevice) { _P_LOCAL()(value, attrib, srcDevice, dstDevice); }
CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource) { _P_LOCAL()(resource); }
CUresult cuGraphicsSubResourceGetMappedArray(CUarray *pArray, CUgraphicsResource resource, unsigned int arrayIndex, unsigned int mipLevel) { _P_LOCAL()(pArray, resource, arrayIndex, mipLevel); }
CUresult cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray, CUgraphicsResource resource) { _P_LOCAL()(pMipmappedArray, resource); }
CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *pDevPtr, size_t *pSize, CUgraphicsResource resource) { _P_LOCAL()(pDevPtr, pSize, resource); }
CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource, unsigned int flags) { _P_LOCAL()(resource, flags); }
CUresult cuGraphicsMapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream) { _P_LOCAL()(count, resources, hStream); }
CUresult cuGraphicsUnmapResources(unsigned int count, CUgraphicsResource *resources, CUstream hStream) { _P_LOCAL()(count, resources, hStream); }
CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) { _P_LOCAL()(ppExportTable, pExportTableId); }

}
