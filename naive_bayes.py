#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 07:35:43 2019

@author: ubuntu
"""
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class NaiveBayesDiscrete():
    #初始化
    def __init__(self,lamda=1):
        self.lamda = lamda
        
    #获取相关参数
    def getParams(self,data):
        self.ck_counts = data.label.value_counts()#训练样本中类为ck的数量集合
        self.ck_name = np.array(self.ck_counts.index)#训练样本中类ck名称集合    
        self.DataNum = len(data)#训练样本总数N
        self.CNum = len(self.ck_counts)#类的个数K
        self.DataSet = data
        
    #计算先验概率
    def calPriorProb(self):
        self.ck_PriorProb = []
        for i in range(self.CNum):
            cx_PriorProb = (self.ck_counts[i]+self.lamda)/(self.DataNum+self.CNum*self.lamda)
            self.ck_PriorProb.append(cx_PriorProb)
            
    #计算条件概率
    def calCondProb(self):
        names = locals()#使用动态变量
        self.CondProb = []#存储所有类别的所有特征取值的条件概率
        self.feat_value = []#所有特征取值列表
        
        #对于每一类别的数据集
        for i in range(len(self.ck_name)):
            names['Q%s' % i] = self.DataSet[self.DataSet["label"]==self.ck_name[i]]#按类别划分数据集
            names['ConProbC%s' % i] = []#定义动态变量，表示各类别中所有特征取值的条件概率集合
            feature_arr = self.DataSet.columns.tolist()[0:len(self.DataSet.columns)-1]#获取训练数据集特征集
            
            #对于每一个特征求该特征各个取值的条件概率
            for feature in (feature_arr):
                
                names['Q%s' % feature]=[]#定义动态变量，表示某个类别的某个特征的所有取值条件概率
                
                #对于某个特征的所有可能取值求条件概率
                for value in self.DataSet[feature].value_counts().index.tolist():
                    
                    #生成所有特征取值列表
                    if value not in self.feat_value:#如果这个取值不在列表中，则加入这个取值
                        self.feat_value.append(value)
                        
                    #这里用了拉普拉斯平滑，使得条件概率不会出现0的情况
                    #如果某个类的某个特征取值在训练集上都出现过，则这样计算
                    if value in names['Q%s' % i][feature].value_counts():
                        temp = (names['Q%s' % i][feature].value_counts()[value]+self.lamda)/(names['Q%s' % i][feature].value_counts().sum()+len(names['Q%s' % i][feature].value_counts())*self.lamda)
                    #如果某个类的某个特征取值并未在训练集上出现，为了避免出现0的情况，分子取1(即lamda平滑因子，取1时为拉普拉斯平滑)
                    else:
                        temp = self.lamda/(names['Q%s' % i][feature].value_counts().sum()+len(names['Q%s' % i][feature].value_counts())*self.lamda)
                    
                    #将求得的特征取值条件概率加入列表
                    names['Q%s' % feature].append(temp)
                #将得到的某个类别的某个特征的所有取值条件概率列表加入某个类别中所有特征取值的条件概率集合
                names['ConProbC%s' % i].extend(names['Q%s' % feature])
            #将某个类别中所有特征取值的条件概率集合加入所有类别所有特征取值的条件概率集合
            self.CondProb.append(names['ConProbC%s' % i])
        #将所有特征取值列表也加入所有类别所有特征取值的条件概率集合(后面用来做columns--列索引)
        self.CondProb.append(self.feat_value)
        #用类别名称的集合来生成行索引index
        index = self.ck_name.tolist()
        index.extend(['other'])#此处由于我最后一行是feat_value，后面会删掉，因此在行索引上也多加一个，后面删掉
        #将所有类别所有特征取值的条件概率集合转换为DataFrame格式
        self.CondProb = pd.DataFrame(self.CondProb,columns=self.CondProb[self.CNum],index = index)
        self.CondProb.drop(['other'],inplace = True)
        
        
    #对一个样本进行预测    
    def predict(self,traindata,testdata):
        self.getParams(traindata)#获取参数
        self.calPriorProb()#获取先验概率
        self.calCondProb()#获取条件概率
    
        self.ClassTotalProb = []#初始化各类别总概率列表
        bestprob = -1#初始化最高概率
        bestfeat = ''#初始化最可能类别
    
        for feat in self.ck_name:
            pp = self.ck_PriorProb[self.ck_name.tolist().index(feat)]#pp为先验概率
            cp = 1#初始化条件概率
            for value in self.feat_value:
                if value in testdata.value_counts().index.tolist():
                    cp = cp * self.CondProb[value][feat]#计算各特征取值的条件概率之积
            TotalProb = pp * cp#条件概率之积与先验概率相乘
            self.ClassTotalProb.append(TotalProb)
        #找到最可能类别和其概率    
        for i in range(len(self.ck_name)):
            if self.ClassTotalProb[i] > bestprob:
                bestprob = self.ClassTotalProb[i]
                bestfeat = self.ck_name[i]
        return bestprob,bestfeat
    
    #计算预测准确度
    def getAccuracy(self,traindata,testdata):
        num = 0
        realFeat = testdata.label.tolist()
        for i in range(len(testdata)):
            temp = testdata.iloc[i][0:len(testdata.columns)-1]    
            predProb,predFeat = self.predict(traindata,temp)
            print(predProb,predFeat,realFeat[i])
            if(realFeat[i] == predFeat):
                num = num + 1
        acc = num / float(len(realFeat))*100
        return acc


class NaiveBayesContinuous():
    #获取训练集每个特征的均值和方差以及类标签的取值集合
    def getMeanStdLabel(self,train_data):
        label_counts=train_data.label.value_counts()   # 计算每种标签个数，0:445, 1:246
        label_arr=np.array(label_counts.index)    # 获得标签列表(0,1)
        label_arr.sort()
        #得到除标签外特征数
        num_feature = len(train_data.columns.values) - 1  # 特征个数 = 8
        #按类别划分数据
        names = locals()           # 获得本地变量字典 names.keys() = [self, train_data, label_counts, label_arr, num_feature, names]
        for i in range(len(label_arr)):
            names['c%s' % i] = train_data[train_data["label"]==label_arr[i]]    # c0(453, 9), c1(238, 9)
        #按类别对每个属性求均值和方差 
        c_mean=[]
        c_std=[]
        for j in range(len(label_arr)):  # 循环每一类
            names['mc%s' % j] = []       # 第0类每列特征的平均值mc0(8,), 第一类每列特征的平均值mc1(8,)
            names['sc%s' % j] = []       # 第0类每列特征的方差sc0(), 第一类每列特征的方差sc1()
            for k in range(num_feature):
#                names['mc%s' % j].append(np.mean(names['c%s' % j][k]))
#                names['sc%s' % j].append(np.std(names['c%s' % j][k],ddof=1))
                names['mc%s' % j].append(np.mean(names['c%s' % j]['%s' % k]))
                names['sc%s' % j].append(np.std(names['c%s' % j]['%s' % k], ddof=1))
        for x in range(len(label_arr)):
            c_mean.append(names['mc%s' % x])  # 提取mc0, mc1
            c_std.append(names['sc%s' % x])   # 提取sc0, sc1
            names['arr_c%s' % x] = np.array(names['c%s' % x]) #
        return c_mean,c_std,label_arr
    
    #计算高斯概率密度函数
    def CalcuGaussProb(self,x,mean,stdev):
        exponent = np.exp(-(np.power(x-mean,2))/(2*np.power(stdev,2)))
        GaussProb = (1/(np.sqrt(2*np.pi)*stdev))*exponent
        return GaussProb
    
    #计算连续数据所属类的概率
    def CalcuClassProbCon(self,arr,cx_mean,cx_std):
        cx_probabilities=1
        for i in range(len(cx_mean)):
            cx_probabilities *= self.CalcuGaussProb(arr[i],cx_mean[i],cx_std[i])
        return cx_probabilities
    
    #单一样本预测
    def predict(self,trainData,testData):
        prob = []
        #print(trainData)
        self.cmean,self.cstd,self.label_array=self.getMeanStdLabel(trainData)
        for i in range(len(self.cmean)):
            cx_mean = self.cmean[i] #x类的均值
            cx_std = self.cstd[i] #x类的方差
            #print(testData)
            prob.append(self.CalcuClassProbCon(testData,cx_mean,cx_std)) #将计算得到的各类别概率存入列表
        bestLabel,bestProb = None,-1 #初始化最可能的类和最大概率值    
        for i in range(len(prob)): #找到所有类别中概率值最大的类
            if prob[i] > bestProb:
                bestProb = prob[i]
                bestLabel = self.label_array[i]
        return bestLabel,bestProb
    
    #整个数据集预测
    def getPredictions(self,TrainData,TestData):
        self.prediction = []
        self.testdata = np.array(TestData)
        for i in range(len(self.testdata)):
            result,proby = self.predict(TrainData,self.testdata[i])
            self.prediction.append(result)
        return self.prediction
    
    #计算准确性
    def getAccuracy(self):
        correct = 0
        for i in range(len(self.testdata)):
            if(self.testdata[i][-1]==self.prediction[i]):
                correct += 1
        return (correct/float(len(self.testdata)))*100.0

            
    
if __name__ == '__main__':
    
    type = 'cont'
    
    if type == 'cont':
        # 连续型数据集： 二分类(0,1), 
        diabetes = pd.read_csv("./dataset/simple/pima_indians_diabetes.csv") # (768, 9)
        dia_train,dia_test = train_test_split(diabetes,test_size=0.1) # (691, 9) + (77, 9)
        model_NBC = NaiveBayesContinuous()
        model_NBC.getPredictions(dia_train,dia_test)
        acc1 = model_NBC.getAccuracy()
        print(model_NBC.prediction)
        print("diabetes predict acc: %.2f" % acc1,"%")
    
    if type == 'disc':
        # 离散型
        car = pd.read_csv("./dataset/simple/car_evalution.csv")
        car_train,car_test = train_test_split(car,test_size=0.1)
        model_NBD = NaiveBayesDiscrete()
        acc2 = model_NBD.getAccuracy(car_train,car_test)
        print("car evalution predict acc: %.2f" % acc2,"%")
    