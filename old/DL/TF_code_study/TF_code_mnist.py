#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:55:22 2018

@author: suliang

以下是照搬TF的mnist.py源代码，并进行初步解析
重点：
    * 如何打开数据和解压数据，判断数据类型
    * 如何做独热编码
    * 如何自定义数据类，并封装数据

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile
from tensorflow.python.util.deprecation import deprecated

# 默认下载地址是从googleeapis下载，所以国内很难下载成功。
# CVDF mirror of http://yann.lecun.com/exdb/mnist/
DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'



# 把解压缩后的数据进行转换
def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

# 对labels进行独热编码
def dense_to_one_hot(labels_dense, num_classes):
"""Convert class labels from scalars to one-hot vectors."""
num_labels = labels_dense.shape[0]
index_offset = numpy.arange(num_labels) * num_classes
labels_one_hot = numpy.zeros((num_labels, num_classes))
labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
return labels_one_hot

# 解压缩子程序：对图形文件解压缩并读取到data变量中
# 如果不采用gz压缩文件，直接读取已经解压缩的文件，修改以下代码也可行
def extract_images(f):
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:  # 利用gzip.GzipFile()解压缩
        magic = _read32(bytestream)  # 利用_read32()函数读取读取
        if magic != 2051:   # 判断magic number是否为2051，不是则报错
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                            (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)  # 利用numpy.frombuffer读取所有buffer
        data = data.reshape(num_images, rows, cols, 1)  # 得到一个4维矩阵
        return data

# 解压缩子程序：对标签文件解压缩并读取到label变量中，中间可调用独热编码子程序完成编码
def extract_labels(f, one_hot=False, num_classes=10):
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:   # 利用gzip.GzipFile()解压缩
        magic = _read32(bytestream)
        if magic != 2049:   # 判断magic number是否为2049，不是则报错
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
            num_items = _read32(bytestream)
            buf = bytestream.read(num_items)
            labels = numpy.frombuffer(buf, dtype=numpy.uint8)
            if one_hot:
                return dense_to_one_hot(labels, num_classes)  # 进行独热编码
            return labels



# 创建数据集的类：
class DataSet(object):
  """Container class for a dataset (deprecated).

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  """

  @deprecated(None, 'Please use alternatives such as official/mnist/dataset.py'
              ' from tensorflow/models.')
  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError(
          'Invalid image dtype %r, expected uint8 or float32' % dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

# 这是一个单独子程序，用于按batch size提取数据
def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate(
          (images_rest_part, images_new_part), axis=0), numpy.concatenate(
              (labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]



# 此处是具体的数据读取流程
# 先拼接文件名，然后从目标地址下载文件
# 打开文件，然后调用解压缩文件的子函数(子函数解压缩，判断数据类型，reshape后封装成data, label)
@deprecated(None, 'Please use alternatives such as official/mnist/dataset.py'
            ' from tensorflow/models.')
def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None,
                   source_url=DEFAULT_SOURCE_URL):
  if fake_data:

    def fake():
      return DataSet(
          [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  if not source_url:  # empty string check
    source_url = DEFAULT_SOURCE_URL

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  
  # 下载第一个文件
  local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   source_url + TRAIN_IMAGES)
  with gfile.Open(local_file, 'rb') as f:
    train_images = extract_images(f)
  
    # 下载第二个文件
  local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   source_url + TRAIN_LABELS)
  with gfile.Open(local_file, 'rb') as f:
    train_labels = extract_labels(f, one_hot=one_hot)

  # 下载第三个文件
  local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   source_url + TEST_IMAGES)
  with gfile.Open(local_file, 'rb') as f:
    test_images = extract_images(f)

  # 下载第四个文件
  local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   source_url + TEST_LABELS)
  with gfile.Open(local_file, 'rb') as f:
    test_labels = extract_labels(f, one_hot=one_hot)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError('Validation size should be between 0 and {}. Received: {}.'
                     .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  options = dict(dtype=dtype, reshape=reshape, seed=seed)
  
  # 再次封装成自定义的数据类型Class DataSet： 把train, labels, options都封装一起
  # 并把数据分成3部分，train, validation, test
  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)

  return base.Datasets(train=train, validation=validation, test=test)


# 程序入口：核心调用程序，用于读取数据的开始
@deprecated(None, 'Please use alternatives such as official/mnist/dataset.py'
            ' from tensorflow/models.')
def load_mnist(train_dir='MNIST-data'):
  return read_data_sets(train_dir)
