# Nccl2Allreduce: Alternative Allreduce ops in place of Horovod, support multi-host with multi-device.

### Environment Requirement:
1) Ubuntu 16.04/18.04 (64bit);
2) NVIDIA CUDA >= 9.0;
3) Tensorflow == 1.14.x;
4) Python mpi4py;
5) NCCL2 devel >= 2.0;

### Usage Example:
```sh
from tensorflow.contrib import nccl2_allreduce

...
grads = opt.compute_gradients(loss)
grads = nccl2_allreduce.allreduce(grads)
train_op = opt.apply_gradients(grads)
...
```

### ImageGenerator + Nccl2Allreduce Example:

```sh
# Install GPU-based Tensorflow on Ubuntu:
pip3 install https://github.com/ghostplant/tensorflow-cuda-optimized/releases/download/tf-1.10-linux/tensorflow-1.10_cuda10.0_ubu1604-cp35-cp35m-linux_x86_64.whl

# Install Deps:
apt install libjpeg62-dev python3-mpi4py libopenmpi-dev

# Install ImageGenerator + Nccl2Allreduce Ops:
git clone https://github.com/ghostplant/tf-image-generator
cd tf-image-generator
./tf-image-generator/setup.sh
./tf-nccl2-allreduce/setup.sh

# Test Nccl2Allreduce Example:
mpiexec -np $(ls /dev/nvidia[0-9]* | wc -w) --allow-run-as-root \
  --map-by slot --bind-to none -x NCCL_DEBUG=INFO ./examples/nccl2_example.py
```
