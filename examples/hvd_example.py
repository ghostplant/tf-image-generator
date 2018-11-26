#!/usr/bin/env python3
'''
[EXEC] IF=enp216s0 && mpiexec -H [hosts,..] --allow-run-as-root --map-by slot --bind-to none -x NCCL_DEBUG=INFO \
  --mca oob_tcp_if_include ${IF} --mca btl_tcp_if_include ${IF} -x NCCL_SOCKET_IFNAME=${IF} ./hvd_example.py
'''

import horovod.tensorflow as hvd
import os, sys, warnings, time
import tensorflow as tf
from tensorflow.contrib import image_generator

hvd.init()
device_rank = hvd.rank()
device_name = 'GPU: %%%dd/%%d' % len(str(hvd.size())) % (device_rank + 1, hvd.size())

ncclHvdExist = (0 == os.system('grep -r ncclAllReduce %s/mpi_lib.*.so >/dev/null 2>&1' % os.path.dirname(hvd.__file__)))

if not ncclHvdExist:
  warnings.warn("NCCL is not detected in horovod modules. Recommend to reinstall horovod with:  HOROVOD_GPU_ALLREDUCE=NCCL pip%d install --upgrade --force-reinstall --no-cache-dir horovod" % sys.version_info.major, Warning)
  if hvd.size() > 1:
    exit(1)


synthetic_data = False

batch_size, n_classes = 64, 1001
height, width = 224, 224

if synthetic_data:
  print('Using Synthetic Data as input images ..')
  images = tf.zeros([batch_size, height, width, 3], dtype=tf.float32)
  labels = tf.zeros([batch_size, ], dtype=tf.int32)
  val_images = tf.zeros(images.shape, dtype=tf.float32)
  val_labels = tf.zeros(labels.shape, dtype=tf.int32)
else:
  dataset = os.environ['DATASET'] if 'DATASET' in os.environ else 'flowers'
  print('Using Raw "%s" JPEG images from disk to GPU with ZeroCopy ..' % dataset)
  if not os.path.exists('/tmp/' + dataset):
    print('Downloading Raw "%s" JPEG images ..' % dataset)
    assert 0 == os.system('rm -rf /tmp/%s.part && mkdir -p /tmp/%s.part' % (dataset, dataset))
    assert 0 == os.system('curl -L https://github.com/ghostplant/lite-dnn/releases/download/lite-dataset/images-%s.tar.gz | tar xzvf - -C /tmp/%s.part >/dev/null' % (dataset, dataset))
    assert 0 == os.system('rm -rf /tmp/%s && mv /tmp/%s.part /tmp/%s' % (dataset, dataset, dataset))

  # external models only accept NHWC as implicit standard input format
  images, labels = image_generator.flow_from_directory(directory_url='/tmp/%s/train/' % dataset, image_format='NHWC',
      batch_size=batch_size, target_size=(height, width), logging=True,
      seed=0, rescale=1.0/255, parallel=8, warmup=True)
  val_images, val_labels = image_generator.flow_from_directory(directory_url='/tmp/%s/validate/' % dataset, image_format='NHWC',
      batch_size=batch_size, target_size=(height, width), logging=False,
      seed=0, rescale=1.0/255, parallel=2, warmup=True)


def create_model(images, labels, reuse=None):
  with tf.variable_scope('model', reuse=reuse):
    # Using external models
    from models import resnet_model
    model = resnet_model.create_resnet50_model()
    if device_rank == 0:
      print('Using %d GPU(s) to %s model %s with %d classes..' % (hvd.size(), 'Eval' if reuse else 'Train', model.__class__.__name__, n_classes))
    X, _ = model.build_network(images, nclass=n_classes, image_depth=3, data_format='NCHW', phase_train=True, fp16_vars=False)
    loss = tf.losses.sparse_softmax_cross_entropy(logits=X, labels=labels)
    accuracy_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(X, labels, 5), tf.float32))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels, tf.int64), tf.argmax(X, 1)), tf.float32))
    return loss, accuracy, accuracy_5, tf.trainable_variables()


loss, accuracy, accuracy_5, weights = create_model(images, labels)
val_loss, val_accuracy, val_accuracy_5, _ = create_model(val_images, val_labels, reuse=True)

lr = 0.001
opt = tf.train.RMSPropOptimizer(lr * hvd.size(), decay=1e-6, momentum=0.9)
train_op = hvd.DistributedOptimizer(opt).minimize(loss)

config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list = str(hvd.local_rank())

with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  if device_rank == 0:
    try:
      import numpy as np
      checkpoint_file = './hvd-checkpoint.npy'
      weights_data = np.load(checkpoint_file)
      assign_ops = [tf.assign(symbolic, host) for symbolic, host in zip(weights, weights_data)]
      sess.run(assign_ops)
      print('Try using pre-trained weights on [%s]..' % device_name)
    except Exception as e:
      print('Not using pre-trained weights:', e)

  print('Launch Training on %s..' % device_name)
  sess.run(hvd.broadcast_global_variables(0))
  total_steps, query_per_steps = 10000, 100
  record = time.time()
  for i in range(total_steps):
    sess.run(train_op)
    if i % query_per_steps == 0 or (i + 1) == total_steps:
      during = time.time() - record
      out_loss, top1, top5, out_val_loss, val_top1, val_top5 = sess.run([loss, accuracy, accuracy_5, val_loss, val_accuracy, val_accuracy_5])
      record = time.time()
      print('[%s] step = %d, loss = %.4f, top1 = %3.1f%%, top5 = %3.1f%%, val_loss = %.4f, val_top1 = %3.1f%%, val_top5 = %3.1f%%  (%.2f images/sec)' %
            (device_name, i + 1, out_loss, top1 * 1e2, top5 * 1e2, out_val_loss, val_top1 * 1e2, val_top5 * 1e2, batch_size * query_per_steps * hvd.size() / during))
  if device_rank == 0:
    print('Saving current weights on [%s]..' % device_name)
    np.save(checkpoint_file, sess.run(weights))

print('Done')
