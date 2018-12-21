#!/bin/bash -e

cd $(dirname $0)/src

for PY_VER in 2.6 2.7 3.5 3.6 3.7; do
  DIST=/usr/local/lib/python${PY_VER}

  if [ ! -e ${DIST}/dist-packages/tensorflow/libtensorflow_framework.so ]; then
    echo "[Python ${PY_VER}] Tensorflow is not found, skip."
    continue
  fi

  echo "[Python ${PY_VER}] Tensorflow is found, installing ImageGenerator ops.. "
  rm -rf ${DIST}/dist-packages/tensorflow/contrib/image_generator
  mkdir -p ${DIST}/dist-packages/tensorflow/contrib/image_generator

  USE_ABI=${USE_ABI:-$(python${PY_VER} -c 'import tensorflow as tf; print("\n".join(tf.sysconfig.get_compile_flags()))' | grep _ABI= | awk -F\= '{print $NF}')}

  if ldd ${DIST}/dist-packages/tensorflow/libtensorflow_framework.so | grep libcuda >/dev/null; then
    WITH_CUDA="-DGOOGLE_CUDA -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart"
  fi

  CMD="gcc -pthread -DNDEBUG -g -fwrapv -shared -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC \
    image_generator_ops.cc \
    -o ${DIST}/dist-packages/tensorflow/contrib/image_generator/_lib_ops.so -std=c++11 -fPIC -O2\
    -I${DIST}/dist-packages/tensorflow/include \
    -I${DIST}/dist-packages/tensorflow/include/external/jpeg -I${DIST}/dist-packages/external/jpeg \
    -L${DIST}/dist-packages/tensorflow/ -ltensorflow_framework -l:libjpeg.so.62 \
     ${WITH_CUDA} -I/usr/local \
    -pthread -Wl,-rpath -Wl,--enable-new-dtags -D_GLIBCXX_USE_CXX11_ABI=${USE_ABI:-0}"
  echo "+ $CMD"
  $CMD

  cp __init__.py ${DIST}/dist-packages/tensorflow/contrib/image_generator
  echo "Done."

done

echo "Finish Installation."
