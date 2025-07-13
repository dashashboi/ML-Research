from sklearn.datasets import fetch_openml
from random import randint
import numpy as np
import writingtofiles
#import matplotlib.pyplot as plt
mnist1 = fetch_openml('mnist_784')


x = np.asarray(mnist1.data).T
x = x / 256 
y = np.array(mnist1.target).T

X_train = x[:,:30000]
y_train = y[:30000]

#X_train.shape : (784, 10000)

def randominit():
    #for weights each row shows all weights for every neuron (ie column corresponds to one weight of all neurons)
    #for biases each row shows bias for each column
    w1 = np.random.randn(10,784)
    b1 = np.random.randn(10,1)
    w2 = np.random.randn(10, 10)
    b2 = np.random.randn(10,1)
    wf = np.random.randn(10, 10)
    bf = np.random.randn(10,1)

    return w1, w2, wf, b1, b2, bf


def sigmoid(values, alpha=1):
    values[np.isnan(values)]=0
    return (1/(1 + np.exp(-values/alpha)))

def dsigmoid(z, alpha=1):
    z[np.isnan(z)]=0
    return (alpha*sigmoid(z, 1)*(1-sigmoid(z,1)))

def relu(values):
    values[np.isnan(values)]=0
    return np.maximum(0,values)

def drelu(values):
    values[np.isnan(values)]=0
    return values>0

def softmax(values):
    values[np.isnan(values)]=0
    return np.exp(values) / np.sum(np.exp(values)) ##overflow yahan aarha hai

def one_hot(Y):
    #Y is y_train
    myshape = (Y.size, 10)
    one_hot_y = np.zeros(myshape)
    one_hot_y[np.arange(Y.size), Y.astype(np.int16)] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def fwdprop(w1, w2, wf, b1, b2, bf, X):
    #each element in a column of a repersents one neuron in the picture with index of that column, ie each column has neurons for different pictures
    z1 = w1.dot(X) + b1
    a1 = sigmoid(z1)
    z2 = w2.dot(a1) + b2
    a2 = relu(z2)
    zf = wf.dot(a2) + bf  
    zf = zf - 400*((np.max(zf)//400) +1)
    af = np.apply_along_axis(softmax, 0, zf)
    return z1, a1, z2, a2, zf, af

def backprop(z1, a1, z2, a2, zf, af, wf, w2, X_train, y_train):
    m = y_train.size
    one_hot_y = one_hot(y_train)
    DC_dzf = (af - one_hot_y)
    DC_wf = (1/m) * DC_dzf.dot(a2.T)
    DC_bf = (1/m) * np.sum(DC_dzf, axis=1)
    DC_dz2 = wf.T.dot(DC_dzf) * drelu(z2)
    DC_w2 = (1/m) * DC_dz2.dot(a1.T)
    DC_b2 = (1/m) * np.sum(DC_dz2, axis=1)
    DC_dz1 = w2.T.dot(DC_dz2) * dsigmoid(z1)
    DC_dz1[np.isnan(DC_dz1)] = 0
    DC_w1 = (1/m) * DC_dz1.dot(X_train.T)
    DC_b1 = (1/m) * np.sum(DC_dz1, axis=1)
    
    return DC_w1, DC_w2, DC_wf, DC_b1, DC_b2, DC_bf


def updateparams(w1, b1, w2, b2, wf, bf, dw1, db1, dw2, db2, dwf, dbf, alpha):
    
    dw1[np.isnan(dw1)]=0
    dw2[np.isnan(dw2)]=0
    dwf[np.isnan(dwf)]=0
    db1[np.isnan(db1)]=0
    db2[np.isnan(db2)]=0
    dbf[np.isnan(dbf)]=0

    db1 = np.expand_dims(db1, axis=1)
    db2 = np.expand_dims(db2, axis=1)
    dbf = np.expand_dims(dbf, axis=1)

    w1 = np.subtract(w1 , alpha*dw1)
    b1 = np.subtract(b1 , alpha*db1)
    w2 = np.subtract(w2 , alpha*dw2)
    b2 = np.subtract(b2 , alpha*db2)
    wf = np.subtract(wf , alpha*dwf)
    bf = np.subtract(bf , alpha*dbf)

    return w1, b1, w2, b2, wf, bf

def pred(af):
    return np.argmax(af,axis=0)

def accuracy(pred, Y):
    print(pred,Y)
    return np.sum(pred == Y.astype(np.int16)) / Y.size

def graddes(X, y, iterations, alpha):
    w1, w2, wf, b1, b2, bf = randominit()
    
    for i in range(iterations):
        z1, a1, z2, a2, zf, af = fwdprop(w1, w2, wf, b1, b2, bf, X)
        dw1, dw2, dwf, db1, db2, dbf = backprop(z1, a1, z2, a2, zf, af, wf, w2, X, y)
        w1, b1, w2, b2, wf, bf = updateparams(w1, b1, w2, b2, wf, bf, dw1, db1, dw2, db2, dwf, dbf, alpha)
        if not(i%10):
            print("iteration: ", i)
            print("accuracy: ", accuracy(pred(af), y))
            writingtofiles.writetofile(w1, w2, wf, b1, b2, bf)
    return w1, b1, w2, b2, wf, bf


w1, b1, w2, b2, wf, bf = graddes(X_train, y_train, 30000, 0.5)












