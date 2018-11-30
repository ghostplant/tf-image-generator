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
  rank = int(os.environ['OMPI_COMM_WORLD_RANK']) if 'OMPI_COMM_WORLD_RANK' in os.environ else 0
  size = int(os.environ['OMPI_COMM_WORLD_SIZE']) if 'OMPI_COMM_WORLD_SIZE' in os.environ else 1
  local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']) if size > 1 else 0   # Force to accept OpenMPI only
  return rank, size, local_rank

def broadcast_global_variables(sourceRank=0):
  MPI.COMM_WORLD.Barrier()
  return library.nccl2_broadcast(tf.trainable_variables(), sourceRank=sourceRank)

def allreduce(grad_gv):
  MPI.COMM_WORLD.Barrier()
  grad_g = [g for g, _ in grad_gv]
  grad_v = [v for _, v in grad_gv]
  grad_g = library.nccl2_allreduce(grad_g)
  return zip(grad_g, grad_v)
