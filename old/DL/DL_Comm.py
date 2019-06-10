#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 21:11:20 2018

1. 计算图/flow: 每一个计算就是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系
    * 程序第一阶段：定义计算图
        * 定义每一个计算(即每一个节点)：a=tf.matmul(x, w1), y=tf.matmul(a, w2), y=tf.sigmoid(y)
        * 定义每一个变量(变量=某常量，或变量=占位符)
    * 第二阶段：在session里边运算
        * 初始化所有变量(常量不需要初始化)
        * 运行各个计算：sess.run(w1), sess.run(cross_entropy, feed_dict={x:X, y_:Y})
    * 一共有2类节点：一类是变量，一类是运算，所以1阶段需要先把这两类节点都定义好
    
2. 张量/tensor：可以理解为一个数组，但实际上是tensorflow运算结果的引用，包含(name名字, shape维度, type类型)
    * 张量跟节点代表的计算结果是一一对应的：一个节点代表一个计算，对应一个张量
    * 名字是张量的唯一标识
    * 维度，跟数组的概念一样
    * 类型，最好指定，否则不同类型计算会报错
    * 张量的使用可以直接使用，比如 result = tf.constant([1.0, 2.0], name='a') + tf.constant([2.0,3.0], name='b')
      也可以先定义张量a, b, 然后result = a + b，这样可读性更好。

3. 会话：用来执行定义好的运算，有两种使用会话的方式
    * sess = tf.Session()
      sess.run(...)
      sess.close()
    * with tf.Session() as sess:
        sess.run(...)

4. 神经网络的计算：
    * 可以参考一个可视化工具：http://playground.tensorflow.org/
    * a = x * W
    * y = a * W
    
5. 神经网络参数与变量
    * 神经网络参数初始化：用满足正态分布的随机数初始化神经网络参数是常用方法
    * 常量声明函数 tf.constant([[0.7, 0.9]])
    * 变量声明函数 tf.Variable()
    * 用随机数初始化变量：weights = tf.Variable(tf.random_normal([2,3], stddev = 2)) 均值默认0，标准差为2
    * 用常数初始化变量：bias = tf.Variable(tf.zeros([3]))  用0初始化
    * 用已有变量初始化变量：w2 = tf.Variable(weights.initialized_value()*2.0)

6. 变量初始化
    * init_op = tf.global_variables_initializer()
    * sess.run(init_op)
    
7. 占位符定义的意义：定义占位符，就不用定义常量，也就不用在计算图上增加节点，只需
    * 定义占位符方法
    * feed变量到占位符的方式：
    
8. 线性模型的局限性：

9. tf里边，元素相乘用*， 点积要用tf.matmul()

10. 梯度下降算法是迭代求解最优参数的通用方法，但他只有在损失函数为凸函数才能确保达到全局最优
而其他情况就跟初始值选取很有关系

11. 如何设置学习率：学习率过大，可能导致摇摆而无法收敛，学习率过小会导致迭代时间太长。
tensorflow提供了一种指数衰减法： tf.train.exponential_decay()函数


@author: suliang
"""
import tensorflow as tf

v1 = tf.constant([[1.0, 2.0],[3.0, 4.0]])
v2 = tf.constant([[5.0, 6.0],[7.0, 8.0]])

v3 = tf.Variable(tf.constant(100))

with tf.Session() as sess:
    print(tf.matmul(v1,v2).eval())
    print('-'*20)
    print((v1 * v2).eval())
    print(v1 + v3)


'''
模块1:损失函数——交叉熵的计算（多用于分类问题）
'''
def cr():
    pass


'''
模块2: 损失函数——MSE均方误差的计算（多用于回归问题）
'''
def mse():
    pass


'''
模块3: 多分类的输出概率分布函数softmax
'''
def softmax():
    pass

'''
模块4: MBGD小批量梯度下降算法
'''
def MBGD():
    batch_size = n
    x = tf.placeholder(tf.float32, shape=(batch_size,2), name='x_input')
    y_ = tf.placeholder(tf.float32, shape=(batch_size,1), name='y_input')
    
    loss = 1
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    with tf.Session() as sess:
        w = 1
        for i in range(STEPS):
            current_X, current_Y = (1,2)
            sess.run(train_step, feed_dict={x:current_X, y:current_Y})
    
'''
模块5: 通过集合计算带正则项的损失函数
'''
def get_weight(shape, lambda_):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda_)(var))
    return var

def test():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y = tf.placeholder(tf.float32, shape=(None, 1))
    batch_size = 8
    layer_dimension = [2, 10, 10, 10, 1]
    n_layers = len(layer_dimension)
    
    cur_layer = x
    in_dimension = layer_dimension[0]
    
    for i in range(1, n_layers):
        out_dimension
    
    
    
    

