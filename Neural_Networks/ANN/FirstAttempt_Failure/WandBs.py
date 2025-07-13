#import random
from numpy import array
ws = [r"C:\Users\Hashir Irfan\OneDrive\Desktop\python\files\w1.txt",r"C:\Users\Hashir Irfan\OneDrive\Desktop\python\files\w2.txt",r"C:\Users\Hashir Irfan\OneDrive\Desktop\python\files\wf.txt"]
bs = [r"C:\Users\Hashir Irfan\OneDrive\Desktop\python\files\b1.txt",r"C:\Users\Hashir Irfan\OneDrive\Desktop\python\files\b2.txt",r"C:\Users\Hashir Irfan\OneDrive\Desktop\python\files\b3.txt"]

waw=[[],[],[]]
bab=[]


for i in range(3):
    l1,l2 = [],[]
    with open(ws[i], 'r') as file:
        for line in file:
            l1 = line.strip().split('*')
            l2 = [int(k) for k in l1]
            waw[i].append(l2)

    with open(bs[i], 'r') as file:
        for line in file:
            l1 = line.strip().split('*')
            l2 = [int(k) for k in l1]       
            bab.append(l2)

waw1 = array(waw[0])
waw23 = array([waw[1],waw[2]])
bab = array(bab)

def changebiases(bab,i):   
    mylist= [str(x) for x in bab]
    mystr= "*".join(mylist)

    with open(bs[i],"w") as file:
        file.write(mystr+'\n')

def changeweights(waw, i):
    #send the 2D array only

    loop = 784 if i==0 else 10
    with open(ws[i],"w") as file:
        for j in range(loop):
            mylist= [str(x) for x in waw[j,:]]
            mystr= "*".join(mylist)
            file.write(mystr+'\n')
        
    












