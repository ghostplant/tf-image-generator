from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader
from mpi4py import MPI
import numpy as np

__ops_name__ = __loader__.name.split('.')[-1]

library = loader.load_op_library(resource_loader.get_path_to_datafile('_lib_ops.so'))


def get_node_config():
  comm = MPI.COMM_WORLD
  local = comm.Split_type(MPI.COMM_TYPE_SHARED)
  rank = comm.Get_rank()
  size = comm.Get_size()
  local_rank = local.Get_rank()
  return rank, size, local_rank

def broadcast_global_variables(sourceRank=0):
  MPI.COMM_WORLD.Barrier()
  return library.nccl2_broadcast(tf.trainable_variables(), sourceRank=sourceRank)

def allreduce(grad_gv):
  # MPI.COMM_WORLD.Barrier()
  non_super_grad = []
  inter_param, local_param = 0, 0
  for g, v in grad_gv:
    if 'super_dense_scope' not in v.name:
      non_super_grad.append((g, v))
      inter_param += int(np.product(v.shape))
    else:
      local_param += int(np.product(v.shape))
  print('Total parameter count = %d / inter, %d / local.' % (inter_param, local_param))
  _, size, _ = get_node_config()
  if size == 1:
    return grad_gv
  grad_gv = non_super_grad
  grad_g = [g for g, _ in grad_gv]
  grad_v = [v for _, v in grad_gv]
  grad_g = library.nccl2_allreduce(grad_g)
  return zip(grad_g, grad_v)


@tf.custom_gradient
def super_dense_preprocess(x):
  rank, size, _ = get_node_config()
  def grad(dy):
    if size == 1:
      return dy
    return library.nccl2_allreduce([dy])
  return x, grad

# e.g. y = nccl2_allreduce.super_dense(y, units=2)
def super_dense(data, units, activation=None, kernel_initializer=None, name=None):
  rank, size, _ = get_node_config()
  assert(units % size == 0)
  x = super_dense_preprocess(data)

  with tf.variable_scope('super_dense_scope'):
    y = tf.compat.v1.layers.dense(x, units=units // size, activation=activation, kernel_initializer=kernel_initializer, name=name)

  batch_size = int(data.shape[0])
  if size > 1:
    z = library.super_dense_postprocess(y)
    if batch_size > 1:
      z = tf.transpose(z, [1, 0, 2])
  else:
    z = y
  out = tf.reshape(z, [batch_size, units])
  return out

@tf.RegisterGradient("SuperDensePostprocess")
def _SuperDensePostprocess_grad(unused_op, dy):
  rank, size, _ = get_node_config()
  local_grad = tf.split(dy, [1] * size, axis=0)[rank]
  local_grad = tf.squeeze(local_grad, axis=0)
  return local_grad
