import os

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

from ..tf_base import TFSlimModel

slim = tf.contrib.slim
here = os.path.dirname(os.path.abspath(__file__))

class InceptionV3(TFSlimModel):
# class InceptionV3(object):
  def __init__(self, name=None, num_classes=200, mapping=None):
    self.name = name or "inceptionv3_imagenet"
    self.mapping = mapping
    self.num_classes = num_classes
    if num_classes != 200:
      assert mapping is not None, "If num_classes in the checkpoint is not 200, must supply a label mapping file, in which 200 int numbers separated by space is provided, representing the index of every tinyimagenet synset"
    if self.mapping:
      with open(self.mapping, "r") as f:
        self.label_map = [int(l) for l in f.read().strip().split(" ")]
      assert len(self.label_map) == 200
        
  def __call__(self, inputs, training):
    with slim.arg_scope(inception.inception_v3_arg_scope()):
      inputs = inputs / 255. * 2. - 1.
      self.inputs = inputs = tf.image.resize_images(inputs, (299, 299))
      _, end_points = inception.inception_v3(
        inputs, num_classes=self.num_classes, is_training=training)
      logits = end_points["Logits"]
      if self.mapping:
        logits = tf.gather(logits, self.label_map, axis=-1)
    return logits

Model = InceptionV3

if __name__ == "__main__":
  import sys
  # try 299x299x3 input
  graph = tf.Graph()
  with graph.as_default():
    images = tf.placeholder(tf.float32, (None, 299, 299, 3))
    model = InceptionV3(num_classes=1001, mapping=os.path.join(here, "checkpoints/inception_v3_mapping1001.txt"))
    logits = model(images, training=False)
  model.graph = graph
  model._logits = logits
  model._images = images
  with graph.as_default():
    with tf.variable_scope("utilities"):
      saver = tf.train.Saver(slim.get_model_variables())
      sess = tf.Session(graph=graph)
      saver.restore(sess, os.path.join(here, "checkpoints/inception_v3.ckpt"))

  import numpy as np
  from scipy.misc import imresize, imread
  image_lst = os.listdir(sys.argv[1])
  for image_fname in image_lst:
    image_fpath = os.path.join(sys.argv[1], image_fname)
    img = imread(image_fpath)
    img = imresize(img, (299, 299))
    if img.shape != (299, 299, 3):
      continue
    predict = np.squeeze(sess.run(tf.argmax(model._logits, axis=-1), feed_dict={images: img[np.newaxis]}))
    print("{}: {}".format(image_fname, predict))

