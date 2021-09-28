# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 11:07:39 2020

@author: Administrator
"""


import xlrd
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.svm import *
from sklearn.metrics import classification_report
ddf = pd.read_excel(r'C:\Users\Administrator\Desktop\2017111642_张馨月_统计实验1701\2017111642_预测结果.xls', index=False)
df=xlrd.open_workbook(r'C:\Users\Administrator\Desktop\2017111642_张馨月_统计实验1701\项目工程文件\训练集.xls','rb').sheets()[0]
ddff=xlrd.open_workbook(r'C:\Users\Administrator\Desktop\2017111642_张馨月_统计实验1701\2017111642_预测结果.xls','rb').sheets()[0]

data_list=[]  #文本标题
label_list=[]  #文本类别
for irow in range(1,df.nrows):
    wordcut=jieba.cut(df.cell_value(irow,1),cut_all=False)
    data_list.append(" ".join(wordcut))
    label_list.append(df.cell_value(irow,2))
    print(data_list[-1],'-',label_list[-1])
cv=CountVectorizer()
tftf=TfidfTransformer()
wv=cv.fit_transform(data_list)
tfidf=tftf.fit_transform(wv)
print(tfidf.shape)
print(len(label_list))
#tfidf数据的文本向量集
#ddff
data_list2=[]  #文本标题
for irow in range(1,ddff.nrows):
    wordcut2=jieba.cut(ddff.cell_value(irow,1),cut_all=False)
    data_list2.append(" ".join(wordcut2))
    print(data_list2[-1])




train_data_list,train_label_list=tfidf,label_list

import pickle
from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法包
b = MultinomialNB(alpha=0.001).fit(train_data_list, train_label_list)

#调参
############————##################
from sklearn.model_selection import GridSearchCV
l = [i for i in range(1000)]
li=[]
for i in range(1000):
    li.append(l[i]*0.0001)
    
parameters={'alpha':li}
b = MultinomialNB(alpha=0.001)
b_grid=GridSearchCV(b,parameters,cv=5)
b_grid.fit(train_data_list, train_label_list)
#在拟合模型后，可以查看格子搜索的结果
print(b_grid.best_params_)
print(b_grid.best_score_)
##################————#################
#预测
b = MultinomialNB(alpha=0.0979).fit(train_data_list, train_label_list)

new=[]
lei=[]
new_tfidf=[]
for i in range(0,1295):
    a=" ".join(jieba.cut(data_list2[i]))
    new.append(a)
    wvect=CountVectorizer(vocabulary=cv.vocabulary_)
    new_tfidf.append(tftf.fit_transform(wvect.fit_transform([new[i]],)))
    m=str(b.predict(new_tfidf[i]))#####
#    print(m.replace('[','').replace(']','').replace("'",'').replace("'",''))
    c=m.replace('[','').replace(']','').replace("'",'').replace("'",'')
    lei.append(c)
    
from pandas.core.frame import DataFrame
leie=DataFrame(lei)    
a=leie[0]

ddf["类别"] = list(a)
ddf.to_excel(r'C:\Users\Administrator\Desktop\2017111642_张馨月_统计实验1701\2017111642_预测结果.xls', index=False)

