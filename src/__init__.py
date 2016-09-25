from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

__ops_name__ = __loader__.name.split('.')[-1]

library = loader.load_op_library(resource_loader.get_path_to_datafile('_lib_ops.so'))


def image_pipe(directory_url, batch_size, height, width, image_format='NCHW',
               parallel=8, rescale=0.00392157, seed=0, synchronize=True, logging=True, cache_mbytes=512):
  ''' Wrapper of ImagePipe Kernel Op'''

  images, labels = library.image_pipe(
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
    cache_mbytes=cache_mbytes)
  if image_format == 'NCHW':
    images = tf.reshape(images, [batch_size, 3, height, width])
  else:
    images = tf.reshape(images, [batch_size, height, width, 3])

  labels = tf.reshape(labels, [batch_size,])
  return images, labels
