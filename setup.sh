#!/bin/bash -e

cd $(dirname $0)/src

for PY_VER in 2.6 2.7 3.5 3.6 3.7; do
  DIST=/usr/local/lib/python${PY_VER}

  if [ ! -e ${DIST}/dist-packages/tensorflow/libtensorflow_framework.so ]; then
    echo "[Python ${PY_VER}] Tensorflow is not found, skip."
    continue
  fi

  echo -n "[Python ${PY_VER}] Tensorflow is found, installing ImagePipe ops.. "
  rm -rf ${DIST}/dist-packages/tensorflow/contrib/image_pipe
  mkdir -p ${DIST}/dist-packages/tensorflow/contrib/image_pipe

  gcc image_pipe_ops.cc -DGOOGLE_CUDA -std=c++14 -shared -fPIC \
    -I${DIST}/dist-packages/tensorflow/include/ \
    -I$(pwd)/include -I/usr/local -I/usr/local/cuda/include \
    -I${DIST}/dist-packages/tensorflow/include/external/jpeg/ \
    -L/usr/local/cuda/lib64 \
    -L${DIST}/dist-packages/tensorflow/ \
    -o ${DIST}/dist-packages/tensorflow/contrib/image_pipe/_lib_ops.so \
    -ltensorflow_framework -lcudart -l:python/_pywrap_tensorflow_internal.so
  cp __init__.py ${DIST}/dist-packages/tensorflow/contrib/image_pipe
  echo "Done."

done

echo "Finish ImagePipe installation."
