## 概率论与数理统计
**为什么要引入随机变量？**
<br>概率论的核心就是研究事件的概率值，也就相当于研究事件X与概率Y的映射关系Y=f(X)。
但之前研究事件的概率，都是以描述的方式定义事件X和事件的概率Y, 而描述的方式显然无法找到这种映射关系f。
<br>而通过引入随机变量，就把事件定义成了一个数字化的变量X，同时把概率定义成变量Y，这样就通过两个数字化变量X,Y能够定义出两者之间的关系f(X)=Y，
也就可以使用任何数学工具来分析这种映射关系了，这就是随机变量引入的价值。

**什么是条件概率，什么是联合概率，两者有什么区别？**


---
## PART 1. knn classifier
**knn的超参数k怎么选择？**
<br>最好选择奇数，防止投票产生平票

---
## PART 2. logistic regression classifier
**逻辑回归是否适合多分类？**
<br>直接的逻辑回归不适合多分类，但有如下三个思路可进行多分类
- 扩展逻辑回归的外延，把-log(p)的p定义成softmax的输出，就可以适合多分类了，也就相当与softmax reg
- 采用一对一的方法：每一个类别之间进行一对一的分类，产生(n_class-1)的阶乘个分类器，然后投票决定预测结果
- 采用一对多的方法


---
## PART 3. softmax regression classifier


---
## PART 4. perceptron classifier



---
## PART 5. svm classifier
**svm是否适合非线性特征？怎么样才能适合？**

**svm是否适合多分类特征？**

**svm如何选择核函数？**

**为什么高斯核函数的sigma越小，模型越复杂？**

**svm的超参数如何选择：C/sigma？**

**svm与logistic回归都可以做线性分类器，两者有什么区别？**
- 对于特征个数n

---
## PART 6. CART classifier


---
## PART 7. naive bayes classifier





