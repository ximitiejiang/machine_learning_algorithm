#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:55:22 2018

@author: suliang

对tensorflow的mnist.py文件进行初步解析

"""

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

# 解压缩子程序：
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


if '__name__' = '__main__':
    # 下载并打开四个文件
    train_dir = ''
      
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
            
    # 下载第1个文件，并解压缩
    local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                       source_url + TRAIN_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
        train_images = extract_images(f)
    # 下载第2个文件，并解压缩
    local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                       source_url + TRAIN_LABELS)
    with gfile.Open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)
    # 下载第3个文件，并解压缩
    local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                       source_url + TEST_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
        test_images = extract_images(f)
    # 下载第4个文件，并解压缩
    local_file = base.maybe_download(TEST_LABELS, train_dir,
                                       source_url + TEST_LABELS)
    with gfile.Open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)        