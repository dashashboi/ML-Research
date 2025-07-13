from sklearn.datasets import fetch_openml
#import pandas as pd
from random import randint
import numpy as np
#import matplotlib.pyplot as plt
mnist1 = fetch_openml('mnist_784')
x = mnist1.data
y = mnist1.target

#learn numpy and then make 2 neural networks, one would be just feeding the image to network directly without chunking (pahle random weights and biases then backprop)
#second: 1st layer will have chunks and second will be fed directly, first with random weights and biases then back prop...
#third layer is output for both

temp = np.asarray(x)
myarr= np.divide(temp, 256)

from WandBs import *
import backprop


a1 = []
a2= []
af=[]
ct=0

def sigmoid(value):
    z = 1/(1 + np.exp(-value))
    return z


#first layer
def firstlayer(image,neuron):
    global a1
    nsum = 0
    myneu = np.multiply(image, waw1[:,neuron])
    nsum=np.sum(myneu) + bab[0,neuron]
    a1.append(sigmoid(nsum))


def secondlayer(layer1, neuron):
    global a2
    nsum = 0
    myneu = np.multiply(layer1, waw23[0,:,neuron])
    nsum=np.sum(myneu) + bab[1,neuron]
    a2.append(sigmoid(nsum))


def outlayer(layer2,neuron):
    global af
    nsum = 0
    myneu = np.multiply(layer2, waw23[1,:,neuron])
    nsum=np.sum(myneu) + bab[2,neuron]
    af.append(sigmoid(nsum))


def start(i):
    global a1, a2, af
    a1 = []
    a2= []
    af=[]
    for j in range(10):
        firstlayer(myarr[i],j)
    for j in range(10):
        secondlayer(a1,j)  
    for j in range(10):
        outlayer(a2,j)

def cost(output, ex):
    diff = np.subtract(output,ex)
    mycost = np.square(diff)
    return np.sum(mycost)


def changevals():
    global a1, a2, af, ct, waw23, waw1, bab

    a1 = []
    a2= []
    af=[]
    ran=randint(0,45000)
    start(ran) 
    expected = [1 if z==int(y[ran]) else 0 for z in range(10)]
    
    for i in range(10):
        changeinwf = backprop.proplastlayer(a2[i],af,expected)
        changeinwf = np.array(changeinwf)
        waw23[1,i,:]= np.subtract(waw23[1,i,:], changeinwf)


    changeinbs = backprop.biasoutput(af, expected)
    changeinbs = np.array(changeinbs)
    bab[2,:] = np.subtract(bab[2,:], changeinbs)


    a1 = []
    a2= []
    af=[]
    ran=randint(0,45000)
    start(ran) 
    expected = [1 if z==int(y[ran]) else 0 for z in range(10)]
    
    for i in range(10):
        changeinw2 = backprop.proplayer2(a1[i],a2,af,expected,waw23[1])
        changeinw2 = np.array(changeinw2)
        waw23[0,i,:]= np.subtract(waw23[0,i,:], changeinwf)

    changeinbs = backprop.biaslayer2( a2, af, waw23[1], expected)
    changeinbs = np.array(changeinbs)
    bab[1,:] = np.subtract(bab[1,:], changeinbs)

    
    a1 = []
    a2= []
    af=[]
    ran=randint(0,45000)
    start(ran) 
    expected = [1 if z==int(y[ran]) else 0 for z in range(10)]
    
    for i in range(784):
        changeinw1 = backprop.proplayer1(myarr[ran,i],a1,a2,af,waw23,expected)
        changeinw1 = np.array(changeinw1)
        waw1[i,:]= np.subtract(waw1[i,:], changeinwf)

    changeinbs = backprop.biaslayer1(a1, a2, af, waw23, expected)
    changeinbs = np.array(changeinbs)
    bab[0,:] = np.subtract(bab[0,:], changeinbs)


def startbackprop():
    global a1, a2, af, ct, bab, waw1, waw23
    
    for count in range(20000):
        ct=0
        for i in range(1500):
            a1 = []
            a2= []
            af=[]
            start(i)
            expected = [1 if z==int(y[i]) else 0 for z in range(10)] 
            ct = ct+ cost(af,expected)
        print(ct/1500)
        if ct/1500 > 0.5 :
            changevals()
            for i in range(3):
                changebiases(bab[i],i)
            changeweights(waw1,0)
            changeweights(waw23[0],1)
            changeweights(waw23[1],2)
        else:
            break



def testit():
    global af
    for i in range(100):
        ran = randint(35001,68000)
        start(ran)
        print(af.index(max(af)), y[ran])

testit()








