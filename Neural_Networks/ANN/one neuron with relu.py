from sklearn.datasets import fetch_openml
from random import randint
import numpy as np

#import matplotlib.pyplot as plt
mnist1 = fetch_openml('mnist_784')


x = np.asarray(mnist1.data).T
x = x / 256 
y = np.array(mnist1.target).T

X_train = x[:,:8000]
y_train = y[:8000]

def randominit():
    #for weights each row shows all weights for every neuron (ie column corresponds to one weight of all neurons)
    #for biases each row shows bias for each column
    w1 = np.random.randn(1,784)
    b1 = np.random.randn(1,1)
    w2 = np.random.randn(10, 1)
    b2 = np.random.randn(10,1)
    return w1, w2, b1, b2

def relu(values):
    values[np.isnan(values)]=0
    return np.maximum(0,values)

def drelu(values):
    values[np.isnan(values)]=0
    return values>0

def softmax(values):
    values[np.isnan(values)]=0
    return np.exp(values) / np.sum(np.exp(values)) 

def one_hot(Y):
    #Y is y_train
    myshape = (Y.size, 10)
    one_hot_y = np.zeros(myshape)
    one_hot_y[np.arange(Y.size), Y.astype(np.int16)] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def fwdprop(w1, w2, b1, b2, X):
    #each element in a column of a repersents one neuron in the picture with index of that column, ie each column has neurons for different pictures
    z1 = w1.dot(X) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    z2 = z2 - 400*((np.max(z2)//400) +1)
    a2 = np.apply_along_axis(softmax, 0, z2)
    return z1, a1, z2, a2

def backprop(w2, a1, z1, a2, X_train, y_train):
    m = y_train.size
    one_hot_y = one_hot(y_train)
    DC_dz2 = (a2 - one_hot_y)
    DC_w2 = (1/m) * DC_dz2.dot(a1.T)
    DC_b2 = (1/m) * np.sum(DC_dz2, axis=1)
    DC_dz1 = w2.T.dot(DC_dz2) * drelu(z1)
    DC_w1 = (1/m) * DC_dz1.dot(X_train.T)
    DC_b1 = (1/m) * np.sum(DC_dz1)
    
    return DC_w1, DC_w2, DC_b1, DC_b2


def updateparams(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    
    dw1[np.isnan(dw1)]=0
    dw2[np.isnan(dw2)]=0
    db2[np.isnan(db2)]=0

    db2 = np.expand_dims(db2, axis=1)

    w1 = np.subtract(w1 , alpha*dw1)
    b1 = np.subtract(b1 , alpha*db1)
    w2 = np.subtract(w2 , alpha*dw2)
    b2 = np.subtract(b2 , alpha*db2)

    return w1, b1, w2, b2



def pred(af):
    return np.argmax(af,axis=0)

def accuracy(pred, Y):
    print(pred,Y)
    return np.sum(pred == Y.astype(np.int16)) / Y.size

def graddes(X, y, iterations, alpha):
    w1, w2, b1, b2 = randominit()
    
    for i in range(iterations):
        z1, a1, z2, a2 = fwdprop(w1, w2, b1, b2, X)
        dw1, dw2, db1, db2 = backprop(w2, a1, z1, a2, X, y)
        w1, b1, w2, b2 = updateparams(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if not(i%10):
            print("iteration: ", i)
            print("accuracy: ", accuracy(pred(a2), y))
    return w1, b1, w2, b2


w1, b1, w2, b2, wf, bf = graddes(X_train, y_train, 30000, 0.65)
