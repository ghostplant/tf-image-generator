/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
// #include "tensorflow/stream_executor/cuda/cuda_stream.h"
#ifndef __HIP_PLATFORM_HCC__
#include <cuda_runtime_api.h>
#include <nccl.h>
#else
#include <hip/hip_runtime_api.h>
#include <rccl.h>

#define cudaSuccess hipSuccess
#define cudaSetDevice hipSetDevice
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaStream_t hipStream_t
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventRecord hipEventRecord
#define cudaEventQuery hipEventQuery
#define cudaEventDestroy hipEventDestroy
#define cudaErrorNotReady hipErrorNotReady
#define cudaEventDisableTiming 0

#endif

#include <dirent.h>
#include <sys/stat.h>
#include <pthread.h>

#include <mpi.h>

#include <memory>
#include <queue>
#include <string>
#include <vector>
#include <unordered_map>


#if !defined(__linux__)
#error "Only Linux platform is supported at the moment (with CUDA)."
#endif

namespace tensorflow {
namespace {

using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


class Nccl2Handle {
 public:
  Nccl2Handle() {
    CHECK_EQ(MPI_SUCCESS, MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    CHECK_EQ(MPI_SUCCESS, MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

    // LOG(INFO) << "Nccl2Handle Initialize: device-rank = " << mpi_rank;
    ncclUniqueId id;
    if (mpi_rank == 0)
      CHECK_EQ(ncclSuccess, ncclGetUniqueId(&id));
    CHECK_EQ(MPI_SUCCESS, MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    CHECK_EQ(ncclSuccess, ncclGroupStart());
    CHECK_EQ(ncclSuccess, ncclCommInitRank(&comm, mpi_size, id, mpi_rank));
    CHECK_EQ(ncclSuccess, ncclGroupEnd());

    dtype = getenv("FP") && atoi(getenv("FP")) == 16 ? ncclHalf : ncclFloat;
  }

  ncclComm_t getHandle() const {
    return comm;
  }

  ~Nccl2Handle() {
    // LOG(INFO) << "Nccl2Handle Destory inter-session communication: device-rank = " << mpi_rank;
    CHECK_EQ(ncclSuccess, ncclCommDestroy(comm));
  }

  ncclDataType_t dtype;

 private:
  int mpi_size, mpi_rank;
  ncclComm_t comm;
};


static shared_ptr<Nccl2Handle> __ncclComm;
static pthread_mutex_t __g_lock = PTHREAD_MUTEX_INITIALIZER;

static shared_ptr<Nccl2Handle> initializeNccl2() {
  pthread_mutex_lock(&__g_lock);
  if (__ncclComm == nullptr)
    __ncclComm = make_shared<Nccl2Handle>();
  pthread_mutex_unlock(&__g_lock);
  return __ncclComm;
}

static void finalizeNccl2() {
  pthread_mutex_lock(&__g_lock);
  if (__ncclComm.use_count() <= 1)
    __ncclComm = nullptr;
  pthread_mutex_unlock(&__g_lock);
}

template <typename Device>
class Nccl2AllreduceOpKernel: public AsyncOpKernel {
 public:
  explicit Nccl2AllreduceOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), ncclComm(initializeNccl2()) {
  }

  ~Nccl2AllreduceOpKernel() {
    ncclComm = nullptr;
    finalizeNccl2();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    // se::Stream* tensor_stream = c->op_device_context()->stream();
    // const cudaStream_t cu_stream = reinterpret_cast<const cudaStream_t>(
    //     ((se::cuda::CUDAStream*)tensor_stream->implementation())->cuda_stream());
    auto GetGpuStream = [](OpKernelContext* context) -> cudaStream_t {
      const cudaStream_t* ptr = CHECK_NOTNULL(
        reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack()));
      return *ptr;
    };
    cudaStream_t cu_stream = GetGpuStream(c);

    for (int i = c->num_inputs() - 1; i >= 0; --i) {
      Tensor* output;
      OP_REQUIRES_OK_ASYNC(c, c->allocate_output(i, c->input(i).shape(), &output), done);
      CHECK_EQ(ncclSuccess, ncclAllReduce((const void*)c->input(i).tensor_data().data(), (void*)output->tensor_data().data(), c->input(i).NumElements(), __ncclComm->dtype, ncclSum, ncclComm->getHandle(), cu_stream));
    }
    done();
  }

 private:
  shared_ptr<Nccl2Handle> ncclComm;
  TF_DISALLOW_COPY_AND_ASSIGN(Nccl2AllreduceOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("Nccl2Allreduce").Device(DEVICE_GPU), Nccl2AllreduceOpKernel<GPUDevice>);

REGISTER_OP("Nccl2Allreduce")
    .Input("tensor: N * T")
    .Output("sum: N * T")
    .Attr("T: {half, float}")
    .Attr("N: int >= 1")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      for (int i = c->num_inputs() - 1; i >= 0; --i)
        c->set_output(i, c->input(i));
      return Status::OK();
    });


template <typename Device>
class Nccl2BroadcastOpKernel: public AsyncOpKernel {
 public:
  explicit Nccl2BroadcastOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), ncclComm(initializeNccl2()) {
    OP_REQUIRES_OK(c, c->GetAttr("sourceRank", &sourceRank));

    initializeNccl2();
  }

  ~Nccl2BroadcastOpKernel() {
    ncclComm = nullptr;
    finalizeNccl2();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    // se::Stream* tensor_stream = c->op_device_context()->stream();
    // const cudaStream_t cu_stream = reinterpret_cast<const cudaStream_t>(
    //     ((se::cuda::CUDAStream*)tensor_stream->implementation())->cuda_stream());
    auto GetGpuStream = [](OpKernelContext* context) -> cudaStream_t {
      const cudaStream_t* ptr = CHECK_NOTNULL(
        reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack()));
      return *ptr;
    };
    cudaStream_t cu_stream = GetGpuStream(c);

    for (int i = c->num_inputs() - 1; i >= 0; --i) {
      CHECK_EQ(ncclSuccess, ncclBroadcast((const void*)c->input(i).tensor_data().data(), (void*)c->input(i).tensor_data().data(), c->input(i).NumElements(), __ncclComm->dtype, sourceRank, ncclComm->getHandle(), cu_stream));
    }
    done();
  }

 private:
  int sourceRank;
  shared_ptr<Nccl2Handle> ncclComm;
  TF_DISALLOW_COPY_AND_ASSIGN(Nccl2BroadcastOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("Nccl2Broadcast").Device(DEVICE_GPU), Nccl2BroadcastOpKernel<GPUDevice>);

REGISTER_OP("Nccl2Broadcast")
    .Input("tensor: N * T")
    .Attr("T: {half, float}")
    .Attr("N: int >= 1")
    .Attr("sourceRank: int")
    .SetIsStateful();
}
}  // namespace tensorflow

