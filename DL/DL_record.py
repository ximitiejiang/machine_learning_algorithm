#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 21:11:20 2018

1. 计算图/flow: 每一个计算就是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系
    * 程序第一阶段：定义每一个计算(即每一个节点)，第二阶段：在session里边运算
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

    

@author: suliang
"""
import tensorflow as tf
a = tf.constant([1,2], name = 'a')

b = tf.constant([2,3], name = 'b')

result = a + b

