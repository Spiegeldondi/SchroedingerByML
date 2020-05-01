import numpy as np
import matplotlib.pyplot as plt

#%%
def genPot(height, width, pos):
    # exceptions noch abklären (zB. width > 127)
    # Extremfälle abklären (width = 0 oder width = 127)
    # bzw. Überschießen: zB. pos = 126, width = 33
    pot = [0]*127
    hill = [height]*width
    pot[pos-int(width/2):int(width/2+1)] = hill
    return pot

#%%
# h = np.random.random()
# w = np.random.randint(16, 32)
# pos = np.random.randint(32, 96)

#%%
# for i in range(10):
#     h = np.random.random()
#     w = np.random.randint(16, 96)
#     pos = np.random.randint(32, 96)
#     pot = genPot(h,w,pos)
#     plt.plot(pot)
    
def genPotNEW(length):
    pot = [0]*length    
    check = False
    
    while check == False:
        posM = np.random.randint(0, len(pot)+1)
        w = np.random.randint(0, len(pot)+1)
        h = np.random.rand()
        if w%2 != 0 and ((posM+int(w/2)) < len(pot)-1) and ((posM-int(w/2)) >= 1):
            a = posM - int(w/2)
            b = posM + 1 + int(w/2)
            pot[a:b] = [h]*w
            check = True
    
    return pot  