#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:18:47 2018

@author: suliang
reference: https://www.cnblogs.com/pinard/p/6056319.html

scikit-learn中决策树的可视化一般需要安装graphviz。主要包括graphviz的安装和python的graphviz插件的安装。

　　　　第一步是安装graphviz。下载地址在：http://www.graphviz.org/。如果你是linux，可以用apt-get或者yum的方法安装。如果是windows，就在官网下载msi文件安装。无论是linux还是windows，装完后都要设置环境变量，将graphviz的bin目录加到PATH，比如我是windows，将C:/Program Files (x86)/Graphviz2.38/bin/加入了PATH

　　　　第二步是安装python插件graphviz： pip install graphviz

　　　　第三步是安装python插件pydotplus。这个没有什么好说的: pip install pydotplus

　　　　这样环境就搭好了，有时候python会很笨，仍然找不到graphviz，这时，可以在代码里面加入这一行：

　　　　os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

　　　　注意后面的路径是你自己的graphviz的bin目录。
"""



from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


def viewTheTree():
    from IPython.display import Image  
    from sklearn import tree
    import pydotplus 
    dot_data = tree.export_graphviz(clf, out_file=None, 
                             feature_names=iris.feature_names,  
                             class_names=iris.target_names,  
                             filled=True, rounded=True,  
                             special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data)  
    Image(graph.create_png()) 


# 导入iris数据集
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target

# 建立模型
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 取水平轴最小最大值外沿
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # 取竖直轴最小最大值外沿
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))  # np.meshgrid()网格化

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # 
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()

