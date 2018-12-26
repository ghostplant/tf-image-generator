#!/bin/bash -ex

WS_DIR=$(dirname $0)
TEMP_DIR=$(mktemp -d)

gcc -shared -fPIC -O2 -std=c++14 ${WS_DIR}/cuda-profiler.cc -ldl -lpthread -I/usr/local/cuda/include -o ${TEMP_DIR}/libcuda.so.1
LD_LIBRARY_PATH=${TEMP_DIR} "${@}"
rm -rf ${TEMP_DIR}

