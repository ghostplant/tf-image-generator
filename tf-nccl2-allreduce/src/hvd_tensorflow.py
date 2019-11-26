from tensorflow.contrib import nccl2_allreduce

device_rank, device_size, device_local_rank = nccl2_allreduce.get_node_config()

def init():
	print('Init MPI node of %d/%d (local_rank = %d)..' % (device_rank, device_size, device_local_rank))

def local_rank():
	return device_local_rank

def size():
	return device_size

def rank():
	return device_rank

def allreduce(grad, average=False, device_dense=''):
	assert(average == False)
	assert(device_dense == '')
	return list(nccl2_allreduce.allreduce([(grad, None)]))[0][0]

def broadcast_global_variables(src=0):
	assert(src == 0)
	return nccl2_allreduce.broadcast_global_variables()
