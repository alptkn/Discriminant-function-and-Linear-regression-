# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:08:57 2019

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# Get data of train and test sets 
points = np.genfromtxt("classification_train.txt",dtype = float,skip_header = 1);
testset = np.genfromtxt("classification_test.txt",dtype = float,skip_header = 1);

x = np.array([])
y = np.array([])
a = np.array([[0.0],[0.0]])
a[0,0] = points[0,0]
a[1,0] = points[0,1]
meanClassOne = np.array([[0.0],[0.0]]) #Mean matrix for class 1
meanClassTwo = np.array([[0.0],[0.0]]) #Mean matrix for class 2
covClassOne = np.array([[0.0,0.0],[0.0,0.0]]) #Covariance matrix for class 1
covClassTwo = np.array([[0.0,0.0],[0.0,0.0]]) #Covariance matrix for class 1
u11 = 0 #sum of data in class 1 feature 1
u12 = 0 #sum of data in class 1 feature 2
u21 = 0 #sum of data in class 2 feature 1
u22 = 0 #sum of data in class 2 feature 1
countclassOne = 0 
countclassTwo = 0

# Finding mean vector of class 1 and class 2
for i in range (0,1600):
    x = np.append(x,float(points[i,0])) #assigne feature 1 to array x
    y = np.append(y,float(points[i,1])) #assigne feature 2 to array y
    if(points[i,2] == 1):
        countclassOne = countclassOne + 1
        u11 = u11 + points[i,0]
        u12 = u12 + points[i,1]
        
    else:
        countclassTwo = countclassTwo + 1
        u21 = u21 + points[i,0]
        u22 = u22 + points[i,1]
Pc1 = float(countclassOne / 1600)
Pc2 = float(countclassTwo / 1600)
plt.scatter(x,y)
plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.show()
print("Probability of class1 {}, probability of class2 {}".format(Pc1,Pc2))
#Mean matrix of class 1
meanClassOne[0,0] = u11 / countclassOne
meanClassOne[1,0] = u12 / countclassOne
#Mean matrix of class2 
meanClassTwo[0,0] = u21 / countclassTwo
meanClassTwo[1,0] = u22 / countclassTwo
print("Mean vector for class1")
print(meanClassOne)
print("Mean vector for class2")
print(meanClassTwo)
#Finding covariance matrix of class 1 and class 2
covone00 = 0.0
covone10 = 0.0
covone11 = 0.0
covtwo00 = 0.0
covtwo10 = 0.0
covtwo11 = 0.0
for i in range(0,1600):
    if(points[i,2] == 1):
        covone00 = covone00 + (points[i,0] - meanClassOne[0,0])**2
        covone10 = covone10 + (points[i,0] - meanClassOne[0,0]) * (points[i,1] - meanClassOne[1,0])
        covone11 = covone11 + (points[i,1] - meanClassOne[1,0])**2
    else:
        covtwo00 = covtwo00 + (points[i,0] - meanClassTwo[0,0])**2
        covtwo10 = covtwo10 + (points[i,0] - meanClassTwo[0,0]) * (points[i,1] - meanClassTwo[1,0])
        covtwo11 = covtwo11 + (points[i,1] - meanClassTwo[1,0])**2
 
#Covariance matrix of class1
covClassOne[0,0] = covone00 / countclassOne;
covClassOne[0,1] = covClassOne[1,0] = covone10 / countclassOne
covClassOne[1,1] = covone11 / countclassOne

#covariance matrix of class2
covClassTwo[0,0] = covtwo00 / countclassTwo
covClassTwo[0,1] = covClassTwo[1,0] = covtwo10 / countclassTwo
covClassTwo[1,1] = covtwo11 / countclassTwo 
print("Covariance matrix for class1")
print(covClassOne)
print("Covariance matrix for class2")
print(covClassTwo)

""" Discriminant Function """
def discriminantFun(x,mean,cov,p):
    inv_covariance = np.linalg.inv(cov) #inverse of covariance matrix 
    mean_tr = np.transpose(mean)        #inverse of mean matrix
    Lnp = np.log(p)
    return(np.dot(np.dot(mean_tr,inv_covariance),x) - (1/2) * (np.dot(np.dot(mean_tr,inv_covariance),mean)) + Lnp)

accuracy = 0.0   
for i in range(0,400):
    a[0,0] = testset[i,0]
    a[1,0] = testset[i,1]
    if(discriminantFun(a,meanClassOne,covClassOne,Pc1) >= discriminantFun(a,meanClassTwo,covClassTwo,Pc2) ):
        if(testset[i,2] == 1):
            accuracy = accuracy + 1
    
    
    if(discriminantFun(a,meanClassTwo,covClassTwo,Pc2) > discriminantFun(a,meanClassOne,covClassOne,Pc1)):
         if(testset[i,2] == 0):
             accuracy = accuracy + 1

accuracyPercentage = (accuracy/400) * 100
print(accuracy)
print(accuracyPercentage)
