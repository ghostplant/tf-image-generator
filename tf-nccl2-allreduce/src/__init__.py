from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader
from mpi4py import MPI

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
  grad_g = [g for g, _ in grad_gv]
  grad_v = [v for _, v in grad_gv]
  grad_g = library.nccl2_allreduce(grad_g)
  return zip(grad_g, grad_v)


@tf.custom_gradient
def super_dense_preprocess(x):
  def grad(dy):
    return library.nccl2_allreduce([dy])
  return x, grad

# e.g. y = nccl2_allreduce.super_dense(y, out_dim=2)
def super_dense(data, out_dim, kernel_initializer=None, name=None):
  rank, size, _ = get_node_config()
  assert(out_dim % size == 0)
  x = super_dense_preprocess(data)

  if name is None:
    import uuid
    name = str(uuid.uuid1())
  partial_w = tf.get_variable("partial_w-" + name, shape=[int(data.shape[1]), out_dim // size], dtype=data.dtype, initializer=kernel_initializer)
  partial_b = tf.get_variable("partial_b-" + name, shape=[1, out_dim // size], dtype=data.dtype, initializer=tf.compat.v1.zeros_initializer())

  y = tf.matmul(x, partial_w)
  y = tf.add(y, partial_b)
  z = library.super_dense_postprocess(y)
  out = tf.reshape(tf.transpose(z, [1, 0, 2]), [int(data.shape[0]), out_dim])
  return out

@tf.RegisterGradient("SuperDensePostprocess")
def _SuperDensePostprocess_grad(unused_op, dy):
  rank, size, _ = get_node_config()
  local_grad = tf.split(dy, [1] * size, axis=0)[rank]
  local_grad = tf.squeeze(local_grad, axis=0)
  return local_grad
