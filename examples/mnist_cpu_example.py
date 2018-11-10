#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.contrib import image_pipe
import os, time

dataset = 'mnist'

if not os.path.exists('/tmp/' + dataset):
  print('Download Raw JPEG images..')
  assert 0 == os.system('rm -rf /tmp/%s.part && mkdir -p /tmp/%s.part' % (dataset, dataset))
  assert 0 == os.system('curl -L https://github.com/ghostplant/lite-dnn/releases/download/lite-dataset/images-%s.tar.gz | tar xzvf - -C /tmp/%s.part >/dev/null' % (dataset, dataset))
  assert 0 == os.system('rm -rf /tmp/%s && mv /tmp/%s.part /tmp/%s' % (dataset, dataset, dataset))

batch_size = 32
height, width = 28, 28

print('Pipeline Raw JPEG images from disk to GPU with ZeroCopy..')
with tf.device('/cpu:0'):
  images, labels = image_pipe.flow_from_directory(directory_url='/tmp/%s/train/' % dataset, image_format='NHWC',
                            batch_size=batch_size, target_size=(height, width), logging=True,
                            seed=0, rescale=1.0/255, parallel=8, warmup=True)
  val_images, val_labels = image_pipe.flow_from_directory(directory_url='/tmp/%s/validate/' % dataset, image_format='NHWC',
                            batch_size=batch_size, target_size=(height, width), logging=False,
                            seed=0, rescale=1.0/255, parallel=2, warmup=True)

tf.set_random_seed(0)

def create_model(images, labels, reuse=None):
  with tf.variable_scope('model', reuse=reuse):
    out = images
    out = tf.layers.conv2d(out, 32, (3, 3), activation=tf.nn.relu, data_format='channels_last')
    out = tf.layers.conv2d(out, 32, (3, 3), activation=tf.nn.relu, data_format='channels_last')
    out = tf.layers.max_pooling2d(out, (2, 2), 2, data_format='channels_last')
    out = tf.layers.dropout(out, 0.25)
    out = tf.layers.conv2d(out, 64, (3, 3), activation=tf.nn.relu, data_format='channels_last')
    out = tf.layers.conv2d(out, 64, (3, 3), activation=tf.nn.relu, data_format='channels_last')
    out = tf.layers.max_pooling2d(out, (2, 2), 2, data_format='channels_last')
    out = tf.layers.dropout(out, 0.25)
    out = tf.layers.flatten(out)
    out = tf.layers.dense(out, 512, activation=tf.nn.relu)
    out = tf.layers.dropout(out, 0.5)
    out = tf.layers.dense(out, 1000)
  loss = tf.losses.sparse_softmax_cross_entropy(logits=out, labels=labels)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels, tf.int64), tf.argmax(out, 1)), tf.float32))
  return loss, accuracy

with tf.device('/cpu:0'):
  loss, accuracy = create_model(images, labels)
  val_loss, val_accuracy = create_model(val_images, val_labels, reuse=True)
  train_op = tf.train.RMSPropOptimizer(0.0001, decay=1e-6).minimize(loss)


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  query_per_steps = 100
  record = time.time()
  for i in range(20000):
    sess.run(train_op)
    if i % query_per_steps == 0:
      during = time.time() - record
      out_loss, acc, val_acc = sess.run([loss, accuracy, val_accuracy])
      record = time.time()
      print('loss = %.4f, accuracy = %.1f%%, val_accuracy = %.1f%%  (%.2f images/sec)' %
               (out_loss, acc * 1e2, val_acc * 1e2, batch_size * query_per_steps / during))
  print('Done.')
