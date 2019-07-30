# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 19:12:17 2019

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
#Getting data from txtx file
points = np.genfromtxt("regression_data.txt",dtype = float,skip_header = 1)/1000;
mse = 0.0
x = np.array([])
y = np.array([])
# Arrays split the data for cross validation 
bx = np.array([])
cx = np.array([])
dx = np.array([])
ex = np.array([])
fx = np.array([])
arrx1 = np.array([])
arrx2 = np.array([])
arrx3 = np.array([])
arrx4 = np.array([])
arrx5 = np.array([])
by = np.array([])
cy = np.array([])
dy = np.array([])
ey = np.array([])
fy = np.array([])
arry1 = np.array([])
arry2 = np.array([])
arry3 = np.array([])
arry4 = np.array([])
arry5 = np.array([])
b_curr = 0.0
m_curr = 0.0
arrx = np.array([])
arryy = np.array([])
#split data to two different array 
for i in range (0,199):
    x = np.append(x,points[i,0])
    y = np.append(y,points[i,1])
    
def gradient_decent(x,y):
    m = b = 0
    iteration = 5000
    n = len(x)
    learning_rate = 0.001
    for i in range(iteration):
        y_guess = m * x + b
        error = (1/n)*sum([val**2 for val in (y-y_guess)])
        md = -(2/n)*sum(x*(y - y_guess)) #partial derivative of m
        bd = -(2/n)*sum(y - y_guess)     #partial derivative of b
        m = m - learning_rate * md
        b = b - learning_rate * bd
    print("Error for gradient decent linear regression {}".format(error))    
    return b,m;
"""Function for mean square error """      
def costFun(b,m,x,y):
    cost = 0.0
    for i in range (len(x)):
        y_guess = m * x[i] + b
        cost = cost + (y[i] - y_guess)**2
    return cost    
             
    
"""5 fold crosss validation """ 
# Split data into 5 array for cross validation 
for i in range (0,199):
    if(i < 40):
        bx = np.append(bx,x[i])
        by = np.append(by,y[i])
    elif (i < 80):
        cx = np.append(cx,x[i])
        cy = np.append(cy,y[i])
    elif (i < 120):
        dx = np.append(dx,x[i])
        dy = np.append(dy,y[i])
    elif (i < 160):
        ex = np.append(ex,x[i])
        ey = np.append(ey,y[i])
    else:
        fx = np.append(fx,x[i])
        fy = np.append(fy,y[i])
learning_rate = 0.001
for i in range (1,6):
    if i == 1:
        arrx1 = np.append(arrx1,cx)
        arrx1 = np.append(arrx1,dx)
        arrx1 = np.append(arrx1,ex)
        arrx1 = np.append(arrx1,fx)
        arry1 = np.append(arry1,cy)
        arry1 = np.append(arry1,dy)
        arry1 = np.append(arry1,ey)
        arry1 = np.append(arry1,fy)
        [b_curr,m_curr] = gradient_decent(arrx1,arry1)
        mse1 = costFun(b_curr,m_curr,bx,by)
        mse1 = mse1 / 40
        print("For 5000 interation b {}, m {}, mse {},learning_rate {}".format(b_curr,m_curr,mse1,learning_rate))
    elif i == 2:
        arrx2 = np.append(arrx2,bx)
        arrx2 = np.append(arrx2,dx)
        arrx2 = np.append(arrx2,ex)
        arrx2 = np.append(arrx2,fx)
        arry2 = np.append(arry2,by)
        arry2 = np.append(arry2,dy)
        arry2 = np.append(arry2,ey)
        arry2 = np.append(arry2,fy)
        [b_curr,m_curr] = gradient_decent(arrx2,arry2)
        mse2 = costFun(b_curr,m_curr,cx,cy)
        mse2 = mse2 / 40
        print("For 5000 interation b {}, m {}, mse {},learning_rate {}".format(b_curr,m_curr,mse2,learning_rate))
    elif i == 3:
        arrx3 = np.append(arrx3,bx)
        arrx3 = np.append(arrx3,cx)
        arrx3 = np.append(arrx3,ex)
        arrx3 = np.append(arrx3,fx)
        arry3 = np.append(arry3,by)
        arry3 = np.append(arry3,cy)
        arry3 = np.append(arry3,ey)
        arry3 = np.append(arry3,fy)
        [b_curr,m_curr] = gradient_decent(arrx3,arry3)
        mse3 = costFun(b_curr,m_curr,dx,dy)
        mse3 = mse3 / 40
        print("For 5000 interation b {}, m {}, mse {},learning_rate {}".format(b_curr,m_curr,mse3,learning_rate))
    elif i == 4:
        arrx4 = np.append(arrx4,bx)
        arrx4 = np.append(arrx4,cx)
        arrx4 = np.append(arrx4,dx)
        arrx4 = np.append(arrx4,fx)
        arry4 = np.append(arry4,by)
        arry4 = np.append(arry4,cy)
        arry4 = np.append(arry4,dy)
        arry4 = np.append(arry4,fy)
        [b_curr,m_curr] = gradient_decent(arrx4,arry4)
        mse4 = costFun(b_curr,m_curr,ex,ey)
        mse4 = mse4 / 40
        print("For 5000 interation b {}, m {}, mse {},learning_rate {}".format(b_curr,m_curr,mse4,learning_rate))
    else:
        arrx5 = np.append(arrx5,bx)
        arrx5 = np.append(arrx5,cx)
        arrx5 = np.append(arrx5,dx)
        arrx5 = np.append(arrx5,ex)
        arry5 = np.append(arry5,by)
        arry5 = np.append(arry5,cy)
        arry5 = np.append(arry5,dy)
        arry5 = np.append(arry5,ey)
        [b_curr,m_curr] = gradient_decent(arrx5,arry5)
        mse5 = costFun(b_curr,m_curr,fx,fy)
        mse5 = mse5 / 39
        print("For 5000 interation b {}, m {}, mse {},learning_rate {}".format(b_curr,m_curr,mse5,learning_rate))

overal_error = (mse1 + mse2 + mse3 + mse4 + mse5) / 5
print(overal_error)        

