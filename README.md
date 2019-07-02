# Machine learning algorithm implementation

Machine learning algorithm implemented by python3: 
trying to build a clear, modular, easy-to-use-and-modify machine learning library. all the machine learning algorithms are rewrited as Class, with same and clear interface. also implement common dataset Class that can be easily used in any algorithms.
As this is a simplified machine learning algorithm implement, the accuracy is not the main factor to be considered, but it can be taken as a baseline, a better acc result is possible to get by optimizing the training hyper params.

### update
- 2019/06/27 add CART algorithm
- 2019/06/26 add naive bayes algorithm
- 2019/06/25 add kdtree algorithm
- 2019/06/21 add svm algorithm
- 2019/06/15 add perceptron algorithm
- 2019/06/14 add softmax regression algorithm
- 2019/06/12 add logistic regression algorithm
- 2019/06/10 add knn regression algorithm
- 2019/06/03 reconstruct this repo

### features
- pure python code to implement all the algorithm.
- all the algorithms integrated as Class, easy to use and modify.
- all the datasets integrated as Class, easy to use and modify.
- all the algorithms are validated on several datasets(sklearn exist datasets or Mnist/Digits).
- support multi class classify by using multi-class-model_wrapper on top of two-class-classify-model.
- support training hyper-parameters modify: batch_size change, learning rate change, model save and load.
- visualization training process: log text and loss curve generation.
- detailed code explanation.

### usage

- prepare main dataset: mnist(from kaggle), other datasets have been prepared by sklearn or in ./dataset/simple/ folder.
```
python3 setup.sh
```
- train(knn/kdtree don't need to train) 
```
from core.softmax_reg_lib import SoftmaxReg
sm = SoftmaxReg(feats, labels)
sm.train()
```
- eval a dataset(support all models)
```
from core.softmax_reg_lib import SoftmaxReg
sm = SoftmaxReg(feats, labels)
sm.load(path='./softmax_reg_weight_2019-5-1_150341.pkl')
sm.evaluation(test_feats, test_labels)
```
- test a sample(support all models)
```
from core.softmax_reg_lib import SoftmaxReg
sm = SoftmaxReg(feats, labels)
sm.load(path='./softmax_reg_weight_2019-5-1_150341.pkl')
sm.predict_single([-1, 8.5])
```
- visualize the linear divide hyperplane(only support logistic_reg/perceptron)
```
from core.softmax_reg_lib import SoftmaxReg
sm = SoftmaxReg(feats, labels)
sm.train()
sm.vis_points_line()
```
- visualize the predict boundary(support all models)
```
from core.softmax_reg_lib import SoftmaxReg
sm = SoftmaxReg(feats, labels)
sm.train()
sm.vis_boundary()
```
- save model(support all models)
```
sm.save('save_folder_path')
```
- load model(support all models)
```
sm.load('model_path')
```

### PART 1.1 knn classifier
<br>feature: 
- no model weight
- support two-classes-classification and multi-classes-classification.
- support linear separable features and nonlinear separable features.
<br>test code: [test_knn](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/test_knn.py).
<br>source code: [knn_reg_lib](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/core/knn_lib.py).
<div align=center><img src="https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/demo/knn.png"/></div>

### PART 1.2 logistic regression classifier
<br>feature: 
- with model weight(n_feat+1, 1).
- only support two-classes-classification.
- support linear separable features.
<br>test code: [test_logistic_reg](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/test_logistic_reg.py).
<br>source code: [logistic_reg_lib](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/core/logistic_reg_lib.py).
<div align=center><img src="https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/demo/logistic.png"/></div>

### PART 1.3 softmax regression classifier
<br>feature: 
- with model weight(n_feat+1, n_class).
- support two-classes-classification and multi-classes-classification.
- support linear separable features.
<br>test code: [test_softmax_reg](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/test_softmax_reg.py).
<br>source code: [softmax_reg_lib](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/core/softmax_reg_lib.py).
<div align=center><img src="https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/demo/softmax.png"/></div>

### PART 1.4 perceptron classifier
<br>feature: 
- with model weight(n_feat+1, 1).
- only support two-classes-classification.
- support linear separable features.
<br>test code: [test_perceptron](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/test_perceptron.py).
<br>source code: [perceptron_lib](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/core/perceptron_lib.py).
<div align=center><img src="https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/demo/perceptron.png"/></div>

### PART 1.5 svm classifier
<br>feature: 
- with model weight.
- only support two-classes-classification.
- support linear separable features and nonlinear separable features.
<br>test code: [test_svm](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/test_svm.py).
<br>source code: [svm_lib](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/core/svm_lib.py).
<div align=center><img src="https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/demo/svm.png"/></div>
<div align=center><img src="https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/demo/svm2.png"/></div>

### PART 1.6 kdtree classifier 
<br>feature:
- no model weight
- support two-classes-classification and multi-classes-classification.
- support linear separable features and nonlinear separable features.
<br>test code: [test_knn](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/test_kdtree.py).
<br>source code: [knn_reg_lib](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/core/kdtree_lib.py).
<div align=center><img src="https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/demo/kdtree.png"/></div>

### PART 1.7 naive bayes classifier 
<br>feature:
- no model weight
- support two-classes-classification and multi-classes-classification.
- support linear separable features and nonlinear separable features(but strongly restricted by features distribution).
- support continuous features and discrete features
<br>test code: [test_cart](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/test_naive_bayes.py).
<br>source code: [cart_lib](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/core/naive_bayes_lib.py).
<div align=center><img src="https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/demo/naive_bayes_continuous.png"/></div>
<div align=center><img src="https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/demo/naive_bayes_continuous2.png"/></div>

### PART 1.8 CART classifier 
<br>feature:
- support two-classes-classification and multi-classes-classification.
- support linear separable features and nonlinear separable features(but strongly restricted by features distribution).
- support continuous features and discrete features
<br>test code: [test_decision_tree](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/test_decision_tree.py).
<br>source code: [decision_tree_lib](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/core/decision_tree_lib.py).
<div align=center><img src="https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/demo/cart.png"/></div>
<div align=center><img src="https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/demo/cart2.png"/></div>

### PART 1.11 random forest classifer
<br>to be update ...

### PART 1.9 MLP classifier(BP network)
<br>to be update ...

### PART 1.10 ada boost classifer
<br>to be update ...

### PART 1.12 OneVSOne model wrapper
<br>feature:
- as a wrapper to transform 2-class classifer to multi-class classifier, can be used on logistic-reg/svm/perceptron 
<br>test code: [test_cart](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/test_ovo_model.py).
<br>source code: [cart_lib](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/core/multi_class_model_wrapper.py).

### PART 1.13 xgboost classifier
<br>to be update ...

### PART 2.1 linear regressor
<br>to be update ...

### PART 3.1 K-means
<br>to be update ...

### PART 4.1 crf
<br>to be update ...

### Reference:
- Machine Learning in Action, Peter Harrington
- Python Machine learning Algorithm, Zhiyong Zhao
- Statical learning method, Hang Li
- https://github.com/WenDesi/lihang_book_algorithm
  

