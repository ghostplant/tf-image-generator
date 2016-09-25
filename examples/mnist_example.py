#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.contrib.image_pipe import image_pipe
import os, time

if not os.path.exists('/tmp/mnist'):
  print('Download Raw JPEG images..')
  assert 0 == os.system('rm -rf /tmp/mnist.part && mkdir -p /tmp/mnist.part')
  assert 0 == os.system('curl -L https://github.com/ghostplant/lite-dnn/releases/download/lite-dataset/images-mnist.tar.gz | tar xzvf - -C /tmp/mnist.part >/dev/null')
  assert 0 == os.system('rm -rf /tmp/mnist && mv /tmp/mnist.part /tmp/mnist')

batch_size = 32

'''
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import OrderedEnqueuer
datagen = ImageDataGenerator(
      data_format='channels_first',
      rescale=1./255,
      horizontal_flip=True,
      rotation_range=10.0,
      zoom_range=0.2,
      width_shift_range=0.1,
      height_shift_range=0.1,
      fill_mode='nearest')

# gen = datagen.flow_from_directory(data_dir, target_size=(32, 32), batch_size=32, class_mode='sparse')
# gen = datagen.flow_from_directory(data_dir, target_size=(32, 32), batch_size=32, class_mode='sparse')
'''

print('Pipeline Raw JPEG images from disk to GPU with ZeroCopy..')
images, labels = image_pipe(directory_url='/tmp/mnist/train/', image_format='NCHW',
                            batch_size=batch_size, height=32, width=32, logging=True,
                            seed=0, rescale=1.0/255, parallel=8)
val_images, val_labels = image_pipe(directory_url='/tmp/mnist/validate/', image_format='NCHW',
                                    batch_size=batch_size, height=32, width=32, logging=False,
                                    seed=0, rescale=1.0/255, parallel=2)

tf.set_random_seed(0)

def create_model(images, labels, reuse=None):
  with tf.variable_scope('model', reuse=reuse):
    out = images
    out = tf.layers.conv2d(out, 32, (3, 3), activation=tf.nn.relu, data_format='channels_first')
    out = tf.layers.conv2d(out, 32, (3, 3), activation=tf.nn.relu, data_format='channels_first')
    out = tf.layers.max_pooling2d(out, (2, 2), 2, data_format='channels_first')
    out = tf.layers.dropout(out, 0.25)
    out = tf.layers.conv2d(out, 64, (3, 3), activation=tf.nn.relu, data_format='channels_first')
    out = tf.layers.conv2d(out, 64, (3, 3), activation=tf.nn.relu, data_format='channels_first')
    out = tf.layers.max_pooling2d(out, (2, 2), 2, data_format='channels_first')
    out = tf.layers.dropout(out, 0.25)
    out = tf.layers.flatten(out)
    out = tf.layers.dense(out, 512, activation=tf.nn.relu)
    out = tf.layers.dropout(out, 0.5)
    out = tf.layers.dense(out, 1000)
  loss = tf.losses.sparse_softmax_cross_entropy(logits=out, labels=labels)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels, tf.int64), tf.argmax(out, 1)), tf.float32))
  return loss, accuracy

loss, accuracy = create_model(images, labels)
val_loss, val_accuracy = create_model(val_images, val_labels, reuse=True)
train_op = tf.train.RMSPropOptimizer(0.0001, decay=1e-6).minimize(loss)


with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  query_per_steps = 200
  record = time.time()
  for i in range(20000):
    sess.run(train_op)
    if (i + 1) % query_per_steps == 0:
      during = time.time() - record
      acc, val_acc = sess.run([accuracy, val_accuracy])
      record = time.time()
      print('performance = %.2f images/sec, accuracy = %.1f%%, val_accuracy = %.1f%%' %
               (batch_size * query_per_steps / during, acc * 1e2, val_acc * 1e2))
  print('Done.')
