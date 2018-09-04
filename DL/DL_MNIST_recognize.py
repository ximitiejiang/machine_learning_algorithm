#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 13:17:51 2018

@author: suliang
教训1: mac上面下载4个数据文件后系统默认都是解压缩的，也就是不带gz后缀
但查看tensorflow源码mnist.py里边read_data_sets()和extract_images()这个文件子函数
发现TF里边默认是要用gz后缀压缩文件的，如果不是压缩文件，读进去的magic number是0x1f8b0808
只有gz文件读进去magic number才是正确的0x0803=2051，源代码说明了如果不等于2051就会报错。
解决办法是：
    * 把自己下载自动解压的文件增加后缀.gz
    * 把自己下载自动解压的文件名修改为跟源码内文件名一致：(.idx3改为-idx3)
    * 把input_data.read_data_sets(path, one_hot=True) 增加False一项，说明为fake data
    从而不用重新下载。

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

BATCH_SIZE = 100   # 每个训练batch中的训练数据个数
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 衰减学习率：指数衰减
REGULARIZATION_RATE = 0.0001  # 正则项系数？？
TRAINING_STEPS = 30000   # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动模型平均衰减率
MODEL_SAVE_PATH="MNIST_model/"  
MODEL_NAME="mnist_model"


# 针对MNIST数据集
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, 
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):

        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], 
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], 
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2



def train(mnist):
    # 创建x,y_
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    
    # 创建正则计算项
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算
    y = inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)


    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, 
                                                          global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, 
                                                                   labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')


    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], 
                                           feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), 
                           global_step=global_step)


def main(argv=None):
    # 导入MNIST数据集，是否可以从其他文件夹导入？
    path = '/Users/suliang/MyDatasets/MNIST/'
    mnist = input_data.read_data_sets(path, one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
    
    
    
