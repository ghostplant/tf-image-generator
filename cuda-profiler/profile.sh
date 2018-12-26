#!/bin/bash -ex

WS_DIR=$(dirname $0)
TEMP_DIR=$(mktemp -d)

gcc -shared -fPIC -O2 -std=c++14 ${WS_DIR}/cuda-profiler.cc -ldl -lpthread -I/usr/local/cuda/include -o ${TEMP_DIR}/libcuda.so.1
LD_LIBRARY_PATH=${TEMP_DIR} "${@}"
rm -rf ${TEMP_DIR}

# vGPU_QUOTA=${vGPU_QUOTA:-50} ./benchmarks/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model resnet50 --batch_size 64 --num_batches 50 --data_format=NCHW --local_parameter_device gpu
