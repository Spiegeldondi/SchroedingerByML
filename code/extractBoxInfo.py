import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae

#%%
path = '/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/'

#%% import potentials
trainx = []
for i in range(20):
    with open(path+'test_pots/test_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainx.append([float(num) for num in row])

#%% example potential to be investigated
boxInfo = []
for potential in trainx:
    pot = potential    
    approx = []
    approxlist = []
    reslist = [2, 3, 5, 8, 13]
    for res in reslist:
        approx = []
        SingleboxInfo = []
        for i in [k for k in range(127)][0::res]:
            minimum = min(pot[i:i+res])
            maximum = max(pot[i:i+res])
            h = (minimum + maximum) / 2
            approx.extend([h]*res)
            if res == 13:
                SingleboxInfo.append(h)
        indx = len(approx)-len(pot) 
        if indx > 0:
            del approx[-indx:] 
        pot = approx
    boxInfo.append(SingleboxInfo)
    
approx4 = []
res = 13 # window size
for i in [k for k in range(127)][0::res]:
    minimum = min(potential[i:i+res])
    maximum = max(potential[i:i+res])
    h = (minimum + maximum) / 2
    approx4.extend([h]*res)
indx = len(approx4)-len(potential) 
del approx4[-indx:]  

plt.plot(potential)
plt.plot(approx, 'r')
plt.plot(approx4)

a = mae(potential, approx)
b = mae(potential, approx4)

print(a)
print(b)
print('error:', abs(b-a)/a*100)