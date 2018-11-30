from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

__ops_name__ = __loader__.name.split('.')[-1]

library = loader.load_op_library(resource_loader.get_path_to_datafile('_lib_ops.so'))


def flow_from_directory(directory_url, batch_size, target_size, image_format='NCHW',
               parallel=8, rescale=1.0, seed=0, synchronize=True, logging=True, cache_mbytes=256, warmup=True):
  ''' Wrapper of ImageGenerator Kernel Op'''

  height, width = target_size

  images, labels = library.image_generator(
    directory_url=directory_url,
    batch_size=batch_size,
    height=height,
    width=width,
    image_format=image_format,
    parallel=parallel,
    rescale=rescale,
    seed=seed,
    synchronize=synchronize,
    logging=logging,
    cache_mbytes=cache_mbytes,
    warmup=warmup)

  if image_format == 'NCHW':
    images = tf.reshape(images, [batch_size, 3, height, width])
  else:
    images = tf.reshape(images, [batch_size, height, width, 3])
  labels = tf.reshape(labels, [batch_size,])

  return images, labels
