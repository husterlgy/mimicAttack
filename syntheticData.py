# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:33:06 2018

@author: husterlgy

Module 1: 用来生成synthetic data的第一个模块。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier as xgbc
from sklearn import preprocessing
import os
import pickle
from collections import Counter
import random
 
#Module 1: 用来生成synthetic data的第一个模块。
def coder_columns(columns,df_date):#把数据中的那些非数值型的变量全部转化成为数值型
    coders = []
    for name in columns:
        coder = preprocessing.LabelEncoder()
        coder.fit(df_date[name].values)
        df_date[name] = coder.transform(list(df_date[name].values))
        coders.append(coder)
    return coders,df_date

#数据加载function
def loadData():
    train_data = pd.read_csv("./data/adult.data", sep = ",", header=None, index_col=False)
    test_data = pd.read_csv("./data/adult.test", sep = ",", header=None, index_col=False)
    #delete extra "."
    test_data[test_data.columns[-1]] = test_data[test_data.columns[-1]].replace([' <=50K.', ' >50K.'],[' <=50K', ' >50K'])
    # Naming the columns :
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country','income']
    train_data.columns = columns
    test_data.columns = columns
    train_data["set"] = "train"
    test_data["set"] =  "test"
    df_date = train_data.append(test_data,ignore_index=True)
    oders,train_data= coder_columns(["workclass","education","marital-status","occupation",
                                                   "relationship","race","sex","native-country","income"],df_date)
    train_data = df_date[df_date["set"] == "train"].reset_index(drop=True)
    test_data = df_date[df_date["set"] == "test"].reset_index(drop=True)
    
    train_data = train_data.drop(["set"],axis=1)
    test_data = test_data.drop(["set"],axis=1)
    print('data loaded!!!')
    return train_data, test_data

#目标模型训练function
def modolTraining(train_data, test_data):
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_xx = train_data[:,0:-1]
    train_yy = train_data[:,-1]
    test_xx = test_data[:,0:-1]
    test_yy = test_data[:,-1]
    #训练模型并保存本地
    if(os.path.exists('./model/xgboost.plk') == False):
        income_clf = xgbc(max_depth=10, n_estimators=150, learning_rate=0.1,min_child_weight=1,seed=27)
        income_clf.fit(train_xx, train_yy)
        train_yy_hat = income_clf.predict(train_xx)
        test_yy_hat = income_clf.predict(test_xx)
        print("Training accuracy:")
        print(accuracy_score(train_yy, train_yy_hat))
        print("Testing accuracy:")
        print(accuracy_score(test_yy, test_yy_hat))    
        with open('./model/xgboost.plk','wb') as f:
            pickle.dump(income_clf, f)
    #加载本地训练模型（income数据最好的test accuracy也就到87%）
    elif(os.path.exists('./model/xgboost.pkl') == True):
        with open('./model/xgboost.pkl','rb') as f:
            income_clf = pickle.load(f)
        train_yy_hat = income_clf.predict(train_xx)
        test_yy_hat = income_clf.predict(test_xx)
        print("Training accuracy:")
        print(accuracy_score(train_yy, train_yy_hat))
        print("Testing accuracy:")
        print(accuracy_score(test_yy, test_yy_hat)) 
    #incomeModel income_clf
    return income_clf
    
#目标模型分析function
#用来分析每一个特征值对分类结果的影响
def featureAnalytics(train_data, test_data, clf_model):
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_xx = train_data[:,0:-1]
    train_yy = train_data[:,-1]
    test_xx = test_data[:,0:-1]
    test_yy = test_data[:,-1]
    classes = list(Counter(train_yy).keys())
    
    #构建训练数据中每一个特征值的可取值范围
    feature_range = {}
    xx = np.vstack([train_xx, test_xx])
    for ii in range(xx.shape[1]):
        feature_value_list = list(Counter(xx[:,ii]).keys())
        feature_range[ii] = np.array(feature_value_list)

    #随机生成一个initial data record,并且每一类class都需要生成一个初始样本
    initial_flag = np.zeros(len(classes))
    initial_record = np.zeros([len(classes), xx.shape[1]])
    while(np.sum(initial_flag) != len(classes)):
        record_temp = np.zeros([1, xx.shape[1]])
        for key in feature_range.keys():
            record_temp[0, key] = np.random.choice(feature_range[key])
        prob_hat = income_clf.predict_proba(record_temp)
        _hat = income_clf.predict(record_temp)
        if((initial_flag[_hat[0]] == 0) and (prob_hat[0,_hat[0]]>0.75)):#如果_hat[0]这一类的初始数据还没合成，并且当前合成的数据预测结果大于阈值，就接收这个合成的数据
            initial_flag[_hat[0]] = 1#修改_hat[0]对应的类的flag值为1
            initial_record[_hat[0], :] = record_temp
    
    #在随机初始化的数据基础上，分析得到被预测成为不同类时，某一个特征值的取值对预测结果proba的影响。
    
            
        
        
        
        
    return initial_record
    
    
    
    
    
    
    






def moduleMain():
    train_data, test_data = loadData()
    income_clf = modolTraining(train_data, test_data)
#    featureAnalytics(train_data, test_data, income_clf)
    
    
    
if __name__ == "__main__":
    moduleMain()
    
    
    
    
    
    
    
    
    
    
    
    
    