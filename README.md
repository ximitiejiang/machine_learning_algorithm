# machine learning algorithm implementation

Machine learning algorithm implemented by python3:
trying to build a clear, modular, easy-to-use-and-modify machine learning library. all the machine learning algorithms are rewrited as Class, with same and clear interface. also implement common dataset Class that can be easily used in any algorithms.

### features
- all the algorithm integrated as Class, easy to use and modify
- all the dataset integrated as Class, easy to use and modify

### usage
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

### Classify
1. KNN: k-nearest neighbors
2. KNNkd: KNN with kd tree - tbf
3. LoR: logistic regression
    * BGA/BGD
    * SGA/SGD
4. SR: softmax regression
5. DT: decision tree(ID3)
6. NB: naive bayes
    * Word2Vec
7. SVM: support vector machine
    * linearKernel
    * polyKernel
    * rbfKernel
    * SMO
8. AD: adaBoost with stump
9. RanF: random forest with CART tree - tbf
10. BPNN

### Regression
11. LiR: linear regression
    * ridge regression - tbf
    * lasso regression - tbf
12. LWLiR: locally weighted linear regression
13. CART regression

### Cluster
14. K-mean
15. DBSCAN

### Deep Learning
16. perception - tbf
17. NN - tbf
18. CNN - tbf


### Reference:
  - Machine Learning in Action, Peter Harrington
  - Python Machine learning Algorithm, Zhiyong Zhao
  

