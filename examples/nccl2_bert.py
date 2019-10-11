#!/usr/bin/env python3
'''
[Local Example]
# mpiexec -np 4 --allow-run-as-root --map-by slot --bind-to none ./nccl2_bert.py

[Cross-node Example]
# IF=enp216s0 && HOSTS=localhost && mpiexec -H ${HOSTS} --allow-run-as-root --map-by slot --bind-to none -x NCCL_DEBUG=INFO \
  --mca oob_tcp_if_include ${IF} --mca btl_tcp_if_include ${IF} -x NCCL_SOCKET_IFNAME=${IF} ./nccl2_bert.py
'''

import os, sys, warnings, time
import tensorflow as tf
import numpy as np

sys.path.append(os.path.dirname(sys.argv[0]) + '/google_bert')

from tensorflow.contrib import nccl2_allreduce
from modeling import BertConfig, BertModel


batch_size, seq_len, nclass = 4, 512, 128
bert_config = BertConfig(
      vocab_size=30522,
      hidden_size=1024, # 768,
      num_hidden_layers=24, # 12,
      num_attention_heads=16, #12,
      intermediate_size=4096, #3072,
      type_vocab_size=2)

input_ids = tf.placeholder(tf.int32, shape=(batch_size, seq_len))
input_mask = tf.placeholder(tf.int32, shape=(batch_size, seq_len))
segment_ids = tf.placeholder(tf.int32, shape=(batch_size, seq_len))
labels = tf.placeholder(tf.int32, shape=(batch_size, ))

feed_dict = {input_ids: None, input_mask: None, segment_ids: None, labels:None}
for x in feed_dict:
  feed_dict[x] = np.ones(x.shape, dtype=x.dtype.as_numpy_dtype())

model = BertModel(
      config=bert_config,
      is_training=False,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=False)

output_layer = model.get_pooled_output()
logits = tf.layers.dense(output_layer, units=nclass, activation=tf.nn.softmax)
loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
variables = tf.trainable_variables()

opt = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
grads = opt.compute_gradients(loss)
grads = nccl2_allreduce.allreduce(grads)
train_op = opt.apply_gradients(grads)


device_rank, device_size, device_local_rank = nccl2_allreduce.get_node_config()
device_name = 'GPU: %%%dd/%%d' % len(str(device_size)) % (device_rank + 1, device_size)

config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list = str(device_local_rank)

with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(nccl2_allreduce.broadcast_global_variables())

  print('Warm up on %s..' % device_name)
  sess.run(train_op, feed_dict=feed_dict)

  print('Launch Training on %s..' % device_name)
  num_steps = 10
  tStart = time.time()
  for i in range(num_steps):
    sess.run(train_op, feed_dict=feed_dict)
  tStop = time.time()
  print('Average perf = %f samples / sec' % (num_steps * batch_size / (tStop - tStart)))

print('Done')
