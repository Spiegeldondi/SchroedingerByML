import numpy as np
import matplotlib.pyplot as plt

#%%    
def genPotNEW(length):
    pot = [0]*length    
    check = False
    
    while check == False:
        posM = np.random.randint(0, len(pot)+1)
        w = np.random.randint(0, len(pot)+1)
        h = np.random.rand()
        if w%2 != 0 and ((posM+int(w/2)) < len(pot)) and ((posM-int(w/2)) >= 0):
            a = posM - int(w/2)
            b = posM + 1 + int(w/2)
            pot[a:b] = [h]*w
            check = True
    
    return pot, a, (b-1) , h

#%%

boxPots = []

for i in range(10):
    boxPots.append(genPotNEW(5))
    plt.plot(boxPots[i][0])
    
plt.plot(boxPots[0][0])