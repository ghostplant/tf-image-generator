#!/usr/bin/env python3
'''
[Local Example]
# mpiexec -np 4 --allow-run-as-root --map-by slot --bind-to none ./nccl2_example.py

[Cross-node Example]
# IF=enp216s0 && HOSTS=localhost && mpiexec -H ${HOSTS} --allow-run-as-root --map-by slot --bind-to none -x NCCL_DEBUG=INFO \
  --mca oob_tcp_if_include ${IF} --mca btl_tcp_if_include ${IF} -x NCCL_SOCKET_IFNAME=${IF} ./nccl2_example.py
'''

import os, sys, warnings, time
import tensorflow as tf
from models import model_config

# support from https://github.com/ghostplant/tf-image-generator
from tensorflow.contrib import image_generator

forward_only = False
using_nccl2 = True

if using_nccl2:
  print('Using Nccl2 allreduce..')
  from tensorflow.contrib import nccl2_allreduce
else:
  print('Not using Nccl2 allreduce..')
  class nccl2_allreduce(object):
    @staticmethod
    def broadcast_global_variables():
        return []
    @staticmethod
    def get_node_config():
        return (0, 1, 0)
    @staticmethod
    def allreduce(grads):
        return grads

using_synthetic_data = False
batch_size, n_classes = 128, 1001
total_steps, query_per_steps = 500, 50
model = model_config.inception_model.Inceptionv3Model()  # Selection of models
# model = model_config.alexnet_model.AlexnetModel()  # Selection of models
image_size = model.get_image_size()

device_rank, device_size, device_local_rank = nccl2_allreduce.get_node_config()
device_name = 'GPU: %%%dd/%%d' % len(str(device_size)) % (device_rank + 1, device_size)


if using_synthetic_data:
  print('Using Synthetic Data as input images ..')
  images = tf.zeros([batch_size, image_size, image_size, 3], dtype=tf.float32)
  labels = tf.zeros([batch_size, ], dtype=tf.int32)
  val_images = tf.zeros(images.shape, dtype=tf.float32)
  val_labels = tf.zeros(labels.shape, dtype=tf.int32)
else:
  dataset = os.environ['DATASET'] if 'DATASET' in os.environ else 'flowers'
  print('Using raw "%s" JPEG images from disk to GPU with ZeroCopy ..' % dataset)
  if not os.path.exists('/tmp/' + dataset):
    print('Downloading "%s" images ..' % dataset)
    assert 0 == os.system('rm -rf /tmp/%s.part && mkdir -p /tmp/%s.part' % (dataset, dataset))
    assert 0 == os.system('curl -L https://github.com/ghostplant/lite-dnn/releases/download/lite-dataset/images-%s.tar.gz | tar xzvf - -C /tmp/%s.part >/dev/null' % (dataset, dataset))
    assert 0 == os.system('rm -rf /tmp/%s && mv /tmp/%s.part /tmp/%s' % (dataset, dataset, dataset))

  images, labels = image_generator.flow_from_directory(directory_url='/tmp/%s/train/' % dataset, image_format='NHWC',
      batch_size=batch_size, target_size=(image_size, image_size), logging=True,
      seed=device_rank, rescale=1.0/255, parallel=8, warmup=True)
  val_images, val_labels = image_generator.flow_from_directory(directory_url='/tmp/%s/validate/' % dataset, image_format='NHWC',
      batch_size=batch_size, target_size=(image_size, image_size), logging=False,
      seed=device_rank, rescale=1.0/255, parallel=2, warmup=True)


def create_model(images, labels, reuse=None):
  with tf.variable_scope('model', reuse=reuse):
    if device_rank == 0:
      print('Using %d GPU(s) to %s model %s with %d classes of size %d.' % (device_size, 'Eval' if reuse else 'Train', model.__class__.__name__, n_classes, image_size))
    X, _ = model.build_network(images, nclass=n_classes, image_depth=3, data_format='NCHW', phase_train=True, fp16_vars=False)
    loss = tf.losses.sparse_softmax_cross_entropy(logits=X, labels=labels)
    accuracy_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(X, labels, 5), tf.float32))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels, tf.int64), tf.argmax(X, 1)), tf.float32))
    return loss, accuracy, accuracy_5, tf.trainable_variables()


loss, accuracy, accuracy_5, weights = create_model(images, labels)
val_loss, val_accuracy, val_accuracy_5, _ = create_model(val_images, val_labels, reuse=True)

lr = 0.00001
opt = tf.train.RMSPropOptimizer(lr, decay=1e-6, momentum=0.9)

grads = opt.compute_gradients(loss)
grads = nccl2_allreduce.allreduce(grads)
train_op = opt.apply_gradients(grads)

config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list = str(device_local_rank)


with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  if device_rank == 0:
    try:
      import numpy as np
      checkpoint_file = './model-checkpoint.npy'
      weights_data = np.load(checkpoint_file, allow_pickle=True)
      assign_ops = [tf.assign(symbolic, host) for symbolic, host in zip(weights, weights_data)]
      sess.run(assign_ops)
      print('Try using pre-trained weights on [%s]..' % device_name)
    except Exception as e:
      print('Not using pre-trained weights:', e)
  sess.run(nccl2_allreduce.broadcast_global_variables())

  print('Launch Training on %s..' % device_name)
  record = time.time()
  avg_val_top1, avg_val_top5 = 0, 0
  for i in range(total_steps):
    if not forward_only:
      sess.run(train_op)
    if forward_only or (i % query_per_steps == 0 or (i + 1) == total_steps):
      during = time.time() - record
      out_loss, top1, top5, out_val_loss, val_top1, val_top5 = sess.run([loss, accuracy, accuracy_5, val_loss, val_accuracy, val_accuracy_5])
      record = time.time()
      print('[%s] step = %d, loss = %.4f, top1 = %3.1f%%, top5 = %3.1f%%, val_loss = %.4f, val_top1 = %3.1f%%, val_top5 = %3.1f%%  (%.2f images/sec)' %
            (device_name, i + 1, out_loss, top1 * 1e2, top5 * 1e2, out_val_loss, val_top1 * 1e2, val_top5 * 1e2, 0.0 if not i else batch_size * query_per_steps * device_size / during))
      if forward_only:
        avg_val_top1 += val_top1
        avg_val_top5 += val_top5
  avg_val_top1 /= total_steps
  avg_val_top5 /= total_steps
  print('Averyage validation top1 = %.2f, top5 = %.2f;' % (avg_val_top1, avg_val_top5))
  if device_rank == 0:
    print('Saving current weights on [%s]..' % device_name)
    np.save(checkpoint_file, sess.run(weights))

print('Done')
