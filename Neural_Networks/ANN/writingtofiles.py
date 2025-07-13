import numpy as np

ws = [r"C:\Users\Hashir Irfan\OneDrive\Desktop\NEURAL NETWORK\files\w1",r"C:\Users\Hashir Irfan\OneDrive\Desktop\NEURAL NETWORK\files\w2",r"C:\Users\Hashir Irfan\OneDrive\Desktop\NEURAL NETWORK\files\wf"]
bs = [r"C:\Users\Hashir Irfan\OneDrive\Desktop\NEURAL NETWORK\files\b1",r"C:\Users\Hashir Irfan\OneDrive\Desktop\NEURAL NETWORK\files\b2",r"C:\Users\Hashir Irfan\OneDrive\Desktop\NEURAL NETWORK\files\bf"]



def writetofile(w1, w2, wf, b1, b2, bf):

    bab = [np.reshape(b1,(10)), np.reshape(b2,(10,)), np.reshape(bf,(10,))]
    waw = np.array([w2, wf])

    for i in range(3):

        bstr = [str(x) for x in bab[i]]
        bstr = "*".join(bstr)

        with open(bs[i],"w") as file:
            file.write(bstr)

        
        with open(ws[i],"w") as file:
            if i == 0:
                for j in range(10):
                    mylist= [str(x) for x in w1[j, :]]
                    mystr= "*".join(mylist)
                    file.write(mystr+'\n')               
            else:    
                for j in range(10):
                    mylist= [str(x) for x in waw[i-1, j, :]]
                    mystr= "*".join(mylist)
                    file.write(mystr+'\n')

















