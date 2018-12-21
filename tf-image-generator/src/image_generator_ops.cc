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
#include "tensorflow/core/platform/posix/posix_file_system.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"

#include <cuda_runtime_api.h>

#include <jpeglib.h>
#include <setjmp.h>

#include <memory>
#include <queue>
#include <string>
#include <vector>
#include <thread>
#include <unordered_map>


#if !defined(__linux__)
#error "Only Linux platform is supported at the moment (with CUDA)."
#endif

namespace tensorflow {
namespace {

using namespace std;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

struct my_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
};

static void jpeg_error_exit(j_common_ptr cinfo) {
  my_error_mgr *myerr = (my_error_mgr*)cinfo->err;
  (*cinfo->err->output_message)(cinfo);
  longjmp(myerr->setjmp_buffer, 1);
}

static bool DecodeImage(const string &path, vector<uint8> &output, int &height_, int &width_, int &depths_) {
  struct jpeg_decompress_struct cinfo;
  FILE * infile;
  JSAMPARRAY buffer;
  int row_stride;

  if ((infile = fopen(path.c_str(), "rb")) == NULL)
    return false;

  struct my_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = jpeg_error_exit;
  if (setjmp(jerr.setjmp_buffer)) {
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return false;
  }

  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, infile);

  (void) jpeg_read_header(&cinfo, TRUE);
  (void) jpeg_start_decompress(&cinfo);
  height_ = cinfo.output_height;
  width_ = cinfo.output_width;
  depths_ = cinfo.output_components;
  CHECK_EQ(depths_ == 3 || depths_ == 1, true);

  row_stride = cinfo.output_width * cinfo.output_components;
  buffer = (*cinfo.mem->alloc_sarray)
		((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

  output.resize(height_ * width_ * depths_);
  uint8 *hptr = output.data();
  while (cinfo.output_scanline < cinfo.output_height) {
    (void) jpeg_read_scanlines(&cinfo, buffer, 1);
    memcpy(hptr, buffer[0], row_stride);
    hptr += row_stride;
  }
  (void) jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(infile);
  return true;

  /* jpeg::UncompressFlags flags;
  jpeg::Uncompress(input.data(), input_size, flags, nullptr,
    [=, &output, &width_, &height_, &depths_](int width, int height, int depths) -> uint8* {
       output.resize(width * height * depths);
       width_ = width, height_ = height, depths_ = depths;
       return output.data();
  }); */
}


template <typename Device>
class ImageGeneratorOpKernel: public AsyncOpKernel {
 public:

  explicit ImageGeneratorOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("directory_url", &directory_url));
    OP_REQUIRES_OK(c, c->GetAttr("batch_size", &batch_size));
    OP_REQUIRES_OK(c, c->GetAttr("height", &height));
    OP_REQUIRES_OK(c, c->GetAttr("width", &width));
    OP_REQUIRES_OK(c, c->GetAttr("image_format", &image_format));
    OP_REQUIRES_OK(c, c->GetAttr("parallel", &parallel));
    OP_REQUIRES_OK(c, c->GetAttr("seed", &seed));
    OP_REQUIRES_OK(c, c->GetAttr("rescale", &rescale));
    OP_REQUIRES_OK(c, c->GetAttr("synchronize", &synchronize));
    OP_REQUIRES_OK(c, c->GetAttr("logging", &logging));
    OP_REQUIRES_OK(c, c->GetAttr("cache_mbytes", &cache_mbytes));
    OP_REQUIRES_OK(c, c->GetAttr("warmup", &warmup));


    if (directory_url.size() > 0 && directory_url[directory_url.size() - 1] != '/')
      directory_url += '/';

    CHECK_EQ(image_format == "NCHW" || image_format == "NHWC", true);

    threadStop = false;
    samples = 0, iter = 0;

    cache_size = (size_t(cache_mbytes) << 20) / (batch_size * 3 * height * width) / sizeof(float) / parallel;
    cache_size = max(cache_size, 4);

    vector<string> _classes;
    if (pfs.GetChildren(directory_url, &_classes).ok()) {
      for (string &cls_name: _classes) {
        string sub_dir = io::JoinPath(directory_url, cls_name);

        vector<string> _images;
        if (pfs.GetChildren(sub_dir, &_images).ok()) {
          for (string &file: _images) {
            int split = file.find_last_of('.');
            if (split < 0)
              continue;

            string ext_name = file.substr(split + 1);
            transform(ext_name.begin(), ext_name.end(), ext_name.begin(), ::tolower);
            if (ext_name != "jpg" && ext_name != "jpeg")
              continue;
            dict[sub_dir].push_back(file);
          }
        }
      }

      for (auto &it: dict) {
        keyset.push_back(it.first);
        sort(keyset.begin(), keyset.end());
        samples += it.second.size();
      }
      n_class = keyset.size();

      isGpuDevice = !!c->device()->tensorflow_gpu_device_info();

      if (logging) {
        LOG(INFO) << "Device for Image Buffers: " << (isGpuDevice ? "GPU" : "CPU (not recommended)");
        LOG(INFO) << "Total images: " << samples <<", belonging to " << n_class << " classes, loaded from '" << directory_url << "';";
        for (int i = 0; i < n_class; ++i)
          LOG(INFO) << "  [*] class-id " << i << " => " << keyset[i] << " with " << dict[keyset[i]].size() << " samples included;";
      }
    }

    if (samples == 0) {
      LOG(FATAL) << "No valid images found in directory '" << directory_url << "'.";
    }

    auto* gpu_info = c->device()->tensorflow_gpu_device_info();
    int gpu_id = isGpuDevice ? gpu_info->gpu_id : -1;

    workers.resize(parallel);
    for (int i = 0; i < parallel; ++i) {
      workers[i].handle = new std::thread([this, i, gpu_id] {
        this->BackgroundWorker(i, gpu_id);
      });
    }

    if (warmup) {
      if (logging)
        LOG(INFO) << "Warming up image generation..";
      for (int i = 0; i < parallel; ++i)
        while (!workers[i].postWarmup)
          usleep(500000);
      if (logging)
        LOG(INFO) << "Finish image warmup.";
    }
  }

  void BackgroundWorker(int idx, int gpu_id) {
    unsigned int local_seed = seed * parallel + idx;
    auto &worker = workers[idx];

// #if GOOGLE_CUDA
    if (isGpuDevice)
      CHECK_EQ(cudaSuccess, cudaSetDevice(gpu_id));
// #endif
    while (1) {
      while (1) {
        worker.mu_.lock();
        if (worker.ord_que.size() >= cache_size) {
          worker.mu_.unlock();
          worker.postWarmup = true;
          usleep(100000);
          if (threadStop)
            return;
          continue;
        }
        worker.mu_.unlock();
        break;
      }

      size_t image_size = (batch_size * 3 * height * width) * sizeof(float), label_size = batch_size * sizeof(int);
      void *image_label_mem = nullptr;
      {
        mutex_lock l(worker.mu_);
        auto &it = worker.buffers;
        if (it.size()) {
          image_label_mem = it.back();
          it.pop_back();
        }
      }

      if (!image_label_mem) {
        if (isGpuDevice)
          CHECK_EQ(cudaSuccess, cudaMallocHost(&image_label_mem, image_size + label_size));
        else
          CHECK_EQ(true, !!(image_label_mem = malloc(image_size + label_size)));
      }

      float *image_mem = (float*)image_label_mem;
      int *label_mem = (int*)(((char*)image_label_mem) + image_size);

      for (int i = 0; i < batch_size; ++i) {
        float *image_offset = image_mem + i * 3 * height * width;
        int *label_offset = label_mem + i;

        while (1) {
          int label = rand_r(&local_seed) % dict.size();
          auto &files = dict[keyset[label]];
          if (files.size() == 0)
            continue;
          int it = rand_r(&local_seed) % files.size();

          int height_ = 0, width_ = 0, depths_ = 0;
          vector<uint8> output;
          if (!DecodeImage(io::JoinPath(keyset[label], files[it]), output, height_, width_, depths_))
            continue;

          uint8 *image_ptr = output.data();
          vector<int> stride;

          int depth = 3;
          if (image_format == "NCHW")
            stride = {width, 1, width * height};
          else // image_format == "NHWC"
            stride = {width * depth, depth, 1};

          for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
              for (int d = 0; d < depth; ++d) {
                int ih = h * height_ / height, iw = w * width_ / width;
                *(image_offset + h * stride[0] + w * stride[1] + d * stride[2]) =
                    *(image_ptr + ih * width_ * depths_ + iw * depths_ + (depths_ == 3 ? d : 0)) * rescale;
              }
            }
          }
          *label_offset = label;
          break;
        }
      }

      mutex_lock l(worker.mu_);
      worker.ord_que.push(image_label_mem);
    }
  }

  ~ImageGeneratorOpKernel() {
    {
      threadStop = true;
      for (auto &worker: workers) {
        worker.handle->join();
        delete worker.handle;
      }

      while (recycleBufferAsync() > 0)
        ;

      for (auto &worker: workers) {
        while (worker.ord_que.size()) {
          worker.buffers.push_back(worker.ord_que.front());
          worker.ord_que.pop();
        }

        for (auto *buff: worker.buffers) {
          if (isGpuDevice)
            CHECK_EQ(cudaSuccess, cudaFreeHost(buff));
          else
            free(buff);
        }
      }
      workers.clear();
    }
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    void *image_label_mem = nullptr;
    int idx = (iter++) % workers.size();
    auto &worker = workers[idx];
    bool hungry = false;
    while (!image_label_mem) {
      mutex_lock l(worker.mu_);
      if (worker.ord_que.size() == 0) {
        if (!hungry) {
          hungry = true;
          if (logging) {
            LOG(INFO) << "Local GPU is hungry for input, please try increasing the number of parallel workers.";
          }
        }
        continue;
      }
      image_label_mem = worker.ord_que.front();
      worker.ord_que.pop();
    }

    Tensor* image_t = nullptr, *label_t = nullptr;
    auto image_shape = (image_format == "NCHW") ? tensorflow::TensorShape({batch_size, 3, height, width}):
      tensorflow::TensorShape({batch_size, height, width, 3});
    auto label_shape = tensorflow::TensorShape({batch_size});
    OP_REQUIRES_OK_ASYNC(c, c->allocate_output(0, image_shape, &image_t), done);
    OP_REQUIRES_OK_ASYNC(c, c->allocate_output(1, label_shape, &label_t), done);

    size_t image_size = (batch_size * 3 * height * width) * sizeof(float);
    float *image_mem = (float*)image_label_mem;
    int *label_mem = (int*)(((char*)image_label_mem) + image_size);

    if (isGpuDevice) {
      se::Stream* tensor_stream = c->op_device_context()->stream();
      const cudaStream_t cu_stream = reinterpret_cast<const cudaStream_t>(
        ((se::cuda::CUDAStream*)tensor_stream->implementation())->cuda_stream());

      CHECK_EQ(cudaSuccess, cudaMemcpyAsync((void*)image_t->tensor_data().data(), image_mem, image_t->NumElements() * sizeof(float), cudaMemcpyHostToDevice, cu_stream));
      CHECK_EQ(cudaSuccess, cudaMemcpyAsync((void*)label_t->tensor_data().data(), label_mem, label_t->NumElements() * sizeof(int), cudaMemcpyHostToDevice, cu_stream));

      if (synchronize) {
        CHECK_EQ(cudaSuccess, cudaStreamSynchronize(cu_stream));

        mutex_lock l(worker.mu_);
        worker.buffers.push_back(image_label_mem);
      } else {
        cudaEvent_t event;
        recycleBufferAsync();
        CHECK_EQ(cudaSuccess, cudaEventCreate(&event));
        CHECK_EQ(cudaSuccess, cudaEventRecord(event, cu_stream));
        lazyRecycleBuffers.push_back({event, image_label_mem, &worker});
      }
    } else {
      memcpy(image_t->flat<float>().data(), image_mem, image_t->NumElements() * sizeof(float));
      memcpy(label_t->flat<int>().data(), label_mem, label_t->NumElements() * sizeof(int));

      mutex_lock l(worker.mu_);
      worker.buffers.push_back(image_label_mem);
    }
    done();
  }

  size_t recycleBufferAsync() {
    for (int i = 0; i < lazyRecycleBuffers.size(); ++i) {
      auto res = cudaEventQuery((cudaEvent_t)lazyRecycleBuffers[i][0]);
      if (res == cudaSuccess) {
        CHECK_EQ(cudaSuccess, cudaEventDestroy((cudaEvent_t)lazyRecycleBuffers[i][0]));
        void *buff = lazyRecycleBuffers[i][1];
        Worker *pWorker = (Worker*)lazyRecycleBuffers[i][2];
        lazyRecycleBuffers[i] = lazyRecycleBuffers.back();
        lazyRecycleBuffers.pop_back();

        mutex_lock l(pWorker->mu_);
        pWorker->buffers.push_back(buff);
        continue;
      }
      CHECK_EQ(res, cudaErrorNotReady);
    }
    return lazyRecycleBuffers.size();
  }

 private:
  unordered_map<string, vector<string>> dict;
  vector<string> keyset;

  struct Worker {
    std::thread *handle;
    mutex mu_;
    queue<void*> ord_que;
    vector<void*> buffers;
    bool postWarmup;

    Worker(): handle(nullptr), postWarmup(false) {
    }
  };

  vector<Worker> workers;
  PosixFileSystem pfs;

  string directory_url, image_format;
  int batch_size, height, width;

  int iter, n_class, samples;
  int cache_size, parallel, seed, cache_mbytes;
  float rescale;

  bool synchronize, logging, warmup, isGpuDevice;
  volatile bool threadStop;

  vector<vector<void*>> lazyRecycleBuffers;

  TF_DISALLOW_COPY_AND_ASSIGN(ImageGeneratorOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("ImageGenerator").Device(DEVICE_GPU), ImageGeneratorOpKernel<GPUDevice>);
REGISTER_KERNEL_BUILDER(Name("ImageGenerator").Device(DEVICE_CPU), ImageGeneratorOpKernel<CPUDevice>);

REGISTER_OP("ImageGenerator")
    .Output("image: float")
    .Output("label: int32")
    .Attr("directory_url: string")
    .Attr("batch_size: int")
    .Attr("height: int")
    .Attr("width: int")
    .Attr("image_format: string = 'NCHW'")
    .Attr("parallel: int = 8")
    .Attr("rescale: float = 0.00392156862")
    .Attr("seed: int = 0")
    .Attr("synchronize: bool = true")
    .Attr("logging: bool = true")
    .Attr("cache_mbytes: int = 256")
    .Attr("warmup: bool = true")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape);

}
}  // namespace tensorflow

