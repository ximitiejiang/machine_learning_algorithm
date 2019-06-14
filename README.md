# Machine learning algorithm implementation

Machine learning algorithm implemented by python3: 
trying to build a clear, modular, easy-to-use-and-modify machine learning library. all the machine learning algorithms are rewrited as Class, with same and clear interface. also implement common dataset Class that can be easily used in any algorithms.
As this is a simplified machine learning algorithm implement, the accuracy is not the main factor to be considered, but it can be take as a baseline, a better acc result is possible to get after training hyper params modification.

### update
- 2019/06/14 add softmax regression algorithm
- 2019/06/12 add logistic regression algorithm
- 2019/06/10 add knn regression algorithm
- 2019/06/03 reconstruct this repo

### features
- all the algorithms integrated as Class, easy to use and modify.
- all the datasets integrated as Class, easy to use and modify.
- all the algorithms are validated on Mnist/Digits.
- support training hyper-parameters modify: batch_size change, learning rate change, model save and load.
- visualization training process: log text and loss curve generation.
- detailed code explanation.

### usage

- prepare dataset: digits(from sklearn), mnist(from kaggle)
```
python3 setup.sh
```
- train: 
```
from core.softmax_reg_lib import SoftmaxReg
sm = SoftmaxReg(feats, labels)
sm.train()
sm.save(root='./')
```
- eval:
```
from core.softmax_reg_lib import SoftmaxReg
sm = SoftmaxReg(feats, labels)
sm.load(path='./softmax_reg_weight_2019-5-1_150341.pkl')
sm.evaluation(test_feats, test_labels)
```
- test a sample
```
from core.softmax_reg_lib import SoftmaxReg
sm = SoftmaxReg(feats, labels)
sm.load(path='./softmax_reg_weight_2019-5-1_150341.pkl')
sm.classify([-1, 8.5])
```
- visualize the divide hyperplane
```
from core.softmax_reg_lib import SoftmaxReg
sm = SoftmaxReg(feats, labels)
sm.train()
sm.vis_points_line()
```

### PART 1. knn classifier

feature: no model weight, support two-classes-classification and multi-classes-classification.
<br>test code: [test_knn](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/test_knn.py)
<br>source code: [knn_reg_lib](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/core/knn_lib.py)

### PART 2. logistic regression classifier

feature: with model weight(n_feat+1, 1), only support two-classes-classification.
<br>test code: [test_logistic_reg](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/test_logistic_reg.py)
<br>source code: [logistic_reg_lib](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/core/logistic_reg_lib.py)

### PART 3. softmax regression classifier

feature: with model weight(n_feat+1, n_class), support two-classes-classification and multi-classes-classification
<br>test code: [test_softmax_reg](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/test_softmax_reg.py)
<br>source code: [softmax_reg_lib](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/core/softmax_reg_lib.py)

### PART 4. perceptron classifier

feature: with model weight(n_feat+1, 1), only support two-classes-classification
<br>test code: [test_perceptron](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/test_perceptron.py)
<br>source code: [perceptron_lib](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/core/perceptron_lib.py)

### PART 5. svm classifier

feature: with model weight
<br>test code: [test_svm](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/test_svm.py)
<br>source code: [svm_lib](https://github.com/ximitiejiang/machine_learning_algorithm/blob/master/core/svm_lib.py)

### Reference:
  - Machine Learning in Action, Peter Harrington
  - Python Machine learning Algorithm, Zhiyong Zhao
  

