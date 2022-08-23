# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 01:05:42 2022

@author: ALIENWARE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
data=pd.read_csv("fma-rock-vs-hiphop.csv")

#%%
bitrate=data.iloc[:,1].values
duration=data.iloc[:,6].values
y=data.iloc[:,8].values
gen=data.iloc[:,9].values
genall=data.iloc[:,10].values
interset=data.iloc[:,12].values
#language=data.iloc[:,13].values
#x=np.array((bitrate,duration,gen,genall,interset,language))
x=np.array((bitrate,duration,gen,genall,interset))
x=np.transpose(x)
#%%
plt.scatter(x[:,0],y,color='red')
plt.title("Raw Data label according to Bitrate")
plt.xlabel('Bitrates')
plt.ylabel(" Lables rook and hip hop")
plt.show()
plt.scatter(y,x[:,1],color='blue')
plt.title("Raw Data label according to Duration")
plt.ylabel('Duration')
plt.xlabel(" Lables rook and hip hop")
plt.show()
plt.scatter(x[:,2],y,color='pink')
plt.title("Raw Data label according to Genres")
plt.xlabel('Genres')
plt.ylabel(" Lables rook and hip hop")
plt.show()
plt.scatter(x[:,3],y,color='green')
plt.title("Raw Data label according to Genres all")
plt.xlabel('Genres all')
plt.ylabel(" Lables rook and hip hop")
plt.show()
plt.scatter(x[:,4],y,color='purple')
plt.title("Raw Data label according to Interest")
plt.xlabel('Interset')
plt.ylabel(" Lables rook and hip hop")
plt.show()
#%%

#PCA ANALYSIS

plt.scatter(x[:,0],y,color='red')
plt.plot(x[:,0],y,color='brown')
plt.title("PCA Data label according to Bitrate")
plt.xlabel('Bitrates')
plt.ylabel(" Lables rook and hip hop")
plt.show()
plt.scatter(y,x[:,1],color='blue')
plt.plot(x[:,1],y,color='pink')
plt.title("PCA Data label according to Duration")
plt.ylabel('Duration')
plt.xlabel(" Lables rook and hip hop")
plt.show()
plt.scatter(x[:,2],y,color='pink')
plt.plot(x[:,2],y,color='purple')
plt.title("PCA Data label according to Genres")
plt.xlabel('Genres')
plt.ylabel(" Lables rook and hip hop")
plt.show()
plt.scatter(x[:,3],y,color='green')
plt.plot(x[:,3],y,color='brown')
plt.title("PCA Data label according to Genres all")
plt.xlabel('Genres all')
plt.ylabel(" Lables rook and hip hop")
plt.show()
plt.scatter(x[:,4],y,color='purple')
plt.plot(x[:,4],y,color='brown')
plt.title("PCA Data label according to Interest")
plt.xlabel('Interset')
plt.ylabel(" Lables rook and hip hop")
plt.show()

#%%
from sklearn.preprocessing import StandardScaler
sd=StandardScaler()
x[:,0:2]=sd.fit_transform(x[:,0:2])
x[:,4:5]=sd.fit_transform(x[:,4:5])
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
#x[:,5]=lb.fit_transform(x[:,5])
#%%
t=[]
value=''
for i in x[:,2]:
    for j in range(1,len(i)):
        if i[j]  !=']':
            value=value+i[j]
        else:
            t.append(value)
            value=''
tr=np.array([t])
tr=np.transpose(tr)
tr=lb.fit_transform(tr)
tr=np.array([tr])
tr=np.transpose(tr)
tr=sd.fit_transform(tr)
x[:,2:3]=tr
#%%
t=[]
value=''
for i in x[:,3]:
    for j in range(1,len(i)):
        if i[j]  !=']':
            value=value+i[j]
        else:
            t.append(value)
            value=''
tr=np.array([t])
tr=np.transpose(tr)
tr=lb.fit_transform(tr)
tr=np.array([tr])
tr=np.transpose(tr)
tr=sd.fit_transform(tr)
x[:,3:4]=tr
#%%
y=lb.fit_transform(y)# 1 for rook 0 for hip hop
#%%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
#%%
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(xtrain,ytrain)
print(lr.score(xtest,ytest)*100)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
y_pred=lr.predict(xtest)
xm=confusion_matrix(ytest, y_pred)
print(xm)
plot_confusion_matrix(lr,xtest,ytest)
#%%
from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier()
dc.fit(xtrain,ytrain)
print(dc.score(xtest,ytest))
xm=confusion_matrix(ytest, y_pred)
print(xm)
plot_confusion_matrix(dc,xtest,ytest)
#%%
from sklearn.svm import SVC
sc=SVC()
sc.fit(xtrain,ytrain)
print(sc.score(xtest,ytest)*100)
xm=confusion_matrix(ytest, y_pred)
print(xm)
plot_confusion_matrix(sc,xtest,ytest)