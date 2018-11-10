# ImagePipe: Image Generator for native Tensorflow, an extremely-fast data input ops flushing data to GPU directly

Usage: Reading original JPEG image directories stored on SSD (in standard Keras directory format), ImagePipe could achieve ~96% performance of synthetic dataset training, which is faster than tf.keras.ImageDataGenerator and more simple than tf.TFRecord (no TF native format conversion needed). Better to work with Horovod for best Distributed Training performance.

### Environment Requirement:
1) Ubuntu 16.04/18.04 (64bit);
2) NVIDIA CUDA >= 9.0;
3) SSD (Recommended);
4) Tensorflow >= 1.10;

### Ops Features:
1) Deterministic image input by configuration of `seed`, which is not supported by tf.keras.preprocessing.ImageDataGenerator;
2) Support direct image generation with either NCHW or NHWC format;
3) Support target image resize in place and interleaving generation;
4) Reference of internal image directory format -
```sh
/train/
    /class-monkey/
        aug_1.jpg
        aug_2.jpg
        ...
    /class-bird/
        aug_1.jpg
        aug_2.jpg
        ...
    ...
```

### The usage of ImagePipe is similar to tf.keras.ImageDataGenerator

### MNIST Example:

```sh
# Install GPU-based Tensorflow on Ubuntu:
pip3 install tensorflow-gpu (>= r1.10)

# Install ImagePipe Ops:
git clone https://github.com/ghostplant/tf-image-pipe
cd tf-image-pipe
./setup.sh

# Test MNIST Example:
./examples/mnist_gpu_example.py
```

### Issues:

If you got error messages like: `tensorflow.python.framework.errors_impl.NotFoundError: /usr/local/lib/python3.5/dist-packages/tensorflow/contrib/image_pipe/_lib_ops.so: undefined symbol: ...`, it means the tensorflow was ever upgraded or downgraded and the ImagePipe ops should be reinstalled by running `./setup.py`.
