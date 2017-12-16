import pandas as pd
import numpy as np
import sklearn
from sklearn import decomposition as skd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import normalize

def loss(y,w, loss = 0): ## cross entropy loss
    for i in range(len(y)):
        loss -= np.sum(np.nan_to_num(np.log(w[i,np.nonzero(y[i])[0][0]] + 0.000000000001)))/len(y)
        
    return loss
def sigmoid_prime(z): ## Derivative of Sigmoid Function
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid(z): ## Activation
    return 1.0/(1.0+np.exp(-z))

def delta_C(y,w): ## rate of change of cost with respect to prob
    A= y/(w+0.1)
    B= (1-y)/(1-(w+.01))
    return (A-B)

def relu_def(x): ## relu activation differentiation
    if x >= 0:
        de = 1
    else :
        de = 0
    return de

relu_prime = np.vectorize(relu_def) ## applying vector wise

def relu(x):
    return np.maximum(x,0)

def activation_sigmoid(p): ## probability for output logits
    li = []
    for i in p:
        z = np.exp(i)
        p = z/(1+ np.sum(z))
        li.append(p)
    pro = np.array(li) 
    return pro




def feed(X,Y,batch_size,cursor = 0):## feeds batches of desired size
    
    while len(X)%batch_size == 0:
        ##to make sure perfect allocation of data
    
        if cursor == len(X)  :
            cursor = 0
    
        x_train = X[cursor:cursor+batch_size]
        y_train = Y[cursor:cursor+batch_size]

    
        cursor += batch_size
        return x_train, y_train

def Neural_network(X,y,weight,bias,step_size, activation):
    
    n = len(weight)
    
    if activation == 1:
            
        
        z_list =[]
        a_list =[]
        for i in range(len(weight)):
            if i == len(weight) - 1:
                z_list.append(np.matmul(a_list[i-1],weight[i]) +bias[i])
                a_list.append(activation_sigmoid(z_list[i]))
            elif i ==0:
                z_list.append(10*normalize((np.matmul(X,weight[i]) +bias[i]), axis = 1, norm = 'max'))
                a_list.append(relu(z_list[i])) 
            else:
                z_list.append(10*normalize((np.matmul(a_list[i-1],weight[i]) +bias[i]), axis = 1, norm = 'max'))
                a_list.append(relu(z_list[i]))
        los = loss(y,a_list[-1], loss = 0)
        print(los)
        dz = []
        dw = []
        db = []
        for i in range(len(weight)):
            if i == 0:
                dz.append(np.multiply(delta_C(y,a_list[-1]) ,  sigmoid_prime(z_list[-1])))
                dw.append((np.matmul(a_list[n-i-2].T,dz[i]))/(np.linalg.norm((np.matmul(a_list[n-i-2].T,dz[i])), axis = 0, ord = 2)))
                db.append(((np.sum(dw[i] , axis = 0, keepdims = False)).T)/np.linalg.norm(((np.sum(dw[i] , axis = 0, keepdims = False)).T), axis = 0, ord = 2))
                
            elif i == len(weight) - 1:
    
                dz.append(np.multiply(np.matmul(dz[-1], weight[n-i].T) , relu_prime(z_list[n-i-1])))
                dw.append((np.matmul(X.T,dz[i]))/(np.linalg.norm((np.matmul(X.T,dz[i])), axis = 0, ord = 2)))
                db.append(((np.sum(dw[i] , axis = 0, keepdims = False)).T)/np.linalg.norm(((np.sum(dw[i] , axis = 0, keepdims = False)).T), axis = 0, ord = 2))
            
            else:
                dz.append(np.multiply(np.matmul(dz[-1], weight[n-i].T) , relu_prime(z_list[n-i-1])))
                dw.append((np.matmul(a_list[n-i-2].T,dz[i]))/(np.linalg.norm((np.matmul(a_list[n-i-2].T,dz[i])), axis = 0, ord = 2)))
                db.append(((np.sum(dw[i] , axis = 0, keepdims = False)).T)/np.linalg.norm(((np.sum(dw[i] , axis = 0, keepdims = False)).T), axis = 0, ord = 2))
    
        dz.reverse()
        dw.reverse()
        db.reverse()
        
        for i in range(len(dz)):
            weight[i] = weight[i] + step_size*dw[i]
            bias[i] = bias[i] + step_size*db[i]

    if activation == 0:
        z_list =[] ## logits
        a_list =[] ## activation
        for i in range(len(weight)):
            if i == len(weight) - 1:
                z_list.append(np.matmul(a_list[i-1],weight[i]) +bias[i])
                a_list.append(activation_sigmoid(z_list[i]))
            elif i ==0:
                z_list.append(10*normalize((np.matmul(X,weight[i]) +bias[i]), axis = 1, norm = 'max'))
                a_list.append(sigmoid(z_list[i])) 
            else:
                z_list.append(10*normalize((np.matmul(a_list[i-1],weight[i]) +bias[i]), axis = 1, norm = 'max'))
                a_list.append(sigmoid(z_list[i]))
        los = loss(y,a_list[-1], loss = 0)
        print(los)
        dz = [] ## rate of change of logits 
        dw = [] ## gradient of weights
        db = [] ## gradient of Biases
        for i in range(len(weight)):
            if i == 0:
                dz.append(np.multiply(delta_C(y,a_list[-1]) ,  sigmoid_prime(z_list[-1])))
                dw.append((np.matmul(a_list[n-i-2].T,dz[i]))/(np.linalg.norm((np.matmul(a_list[n-i-2].T,dz[i])), axis = 0, ord = 2)))
                db.append(((np.sum(dw[i] , axis = 0, keepdims = False)).T)/np.linalg.norm(((np.sum(dw[i] , axis = 0, keepdims = False)).T), axis = 0, ord = 2))
                
            elif i == len(weight) - 1:
        
                dz.append(np.multiply(np.matmul(dz[-1], weight[n-i].T) , sigmoid_prime(z_list[n-i-1])))
                dw.append((np.matmul(X.T,dz[i]))/(np.linalg.norm((np.matmul(X.T,dz[i])), axis = 0, ord = 2)))
                db.append(((np.sum(dw[i] , axis = 0, keepdims = False)).T)/np.linalg.norm(((np.sum(dw[i] , axis = 0, keepdims = False)).T), axis = 0, ord = 2))
            
            else:
                dz.append(np.multiply(np.matmul(dz[-1], weight[n-i].T) , sigmoid_prime(z_list[n-i-1])))
                dw.append((np.matmul(a_list[n-i-2].T,dz[i]))/(np.linalg.norm((np.matmul(a_list[n-i-2].T,dz[i])), axis = 0, ord = 2)))
                db.append(((np.sum(dw[i] , axis = 0, keepdims = False)).T)/np.linalg.norm(((np.sum(dw[i] , axis = 0, keepdims = False)).T), axis = 0, ord = 2))
    
        dz.reverse()
        dw.reverse()
        db.reverse()
        
        for i in range(len(dz)):
            weight[i] = weight[i] + step_size*dw[i]
            bias[i] = bias[i] + step_size*db[i]
    return los, weight,bias