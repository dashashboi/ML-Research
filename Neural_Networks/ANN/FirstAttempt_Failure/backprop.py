
import numpy as np

#logit is inverse of sigmoid, np.log is ln and np.log10 is the normal one
def logit(x):
    if x != 1 or x <= 0:
        return np.log(x/(1-x))
    else:
        return 1


def dsigmoid(z):
    return (np.exp(-z)/((1+np.exp(-z))**2))

#this is for change in wf and bf
#to work with this use with each a2, it will give you dc/dwf for each row of Wf
def proplastlayer( a2, af, exp):
    #a2 is the neuron one of weight in wf is multiplied to get Z...
    #note all other weights in last hidden layer are not tweaked by anyamount hence are constants and dont come in our equation
    #every final output neuron has its own ZL
    #each time it returns a list of all Wx where x is the number of neuron in a2
    dw =[]
    for i in range(10):
        z= logit(af[i])
        dw.append(a2*dsigmoid(z)*2*(af[i]-exp[i]))

    return dw #and db 

#for second hidden layer
#you have to loop all across a1 for this to give you respective rows of weights 
def proplayer2(a1, a2, af, exp, wf):
    #a1 is any neuron in first layer
    #a2 is the entire second layer
    #af is the entire output layer, ie the array in __main__
    #wf is the entire 2D array of final weights
    #dw is for the entire row of weights for the neuron a1
    dw=[]
    
    for i in range(10):
        z2= logit(a2[i])    
        temp=0
        for j in range(10):
            zl= logit(af[j])
            temp = temp + (a1*dsigmoid(z2)*wf[i,j]*dsigmoid(zl)*2*(af[j]-exp[j]))
        dw.append(temp)

    return dw

#for input layer
#repeat 784 times for each pixel to get each row of w1s each time you call it
def proplayer1(pix, a1, a2, af, waw23, exp):
    #pix is any pixel in first layer
    # a1, a2, af are all three complete arrays of layers
    #waw23 are our weights
    #exp is expected array
    
    dw=[]
    for t in range(10):
        z1 = logit(a1[t])
        temp=0
        for i in range(10):
            z2= logit(a2[i])    
            
            for j in range(10):
                zl= logit(af[j])
                temp = temp + (pix*dsigmoid(z1)*waw23[0,t,i]*dsigmoid(z2)*waw23[1,i,j]*dsigmoid(zl)*2*(af[j]-exp[j]))
        dw.append(temp)

    return dw


#for Bf
#returns all change in C wrt biases in final layer
def biasoutput( af, exp):
    db=[]
    for i in range(10):
        z= logit(af[i])
        db.append((dsigmoid(z)*2*(af[i]-exp[i])))
    return db



#for B2s
#returns array of changes we need to make to b2
def biaslayer2(a2,af,wf,exp):
#all of the inputs are full arrays
    db=[]
    for i in range(10):
        temp=0
        z2= logit(a2[i])
        for j in range(10):
            zf= logit(af[j])
            temp = temp + (dsigmoid(z2)*wf[i,j]*dsigmoid(zf)*2*(af[j]-exp[j]))
        db.append(temp)

    return db


#for B1s
#returns array of changes we need to make to b1
def biaslayer1(a1,a2,af,waw23,exp):

    db=[]
    for t in range(10):
        z1= logit(a1[t])
        temp=0        
        for i in range(10):
            z2= logit(a2[i])
            for j in range(10):
                zf= logit(af[j])
                temp = temp + (dsigmoid(z1)*waw23[0,t,i]*dsigmoid(z2)*waw23[1,i,j]*dsigmoid(zf)*2*(af[j]-exp[j]))
        db.append(temp)

    return db















