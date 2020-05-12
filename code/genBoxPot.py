import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#%%
path = '/home/domi/Dokumente/'

#%%
def generatepot(length):
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
    
    return pot, a, (b-1), h

#%%
seed = 42

np.random.seed(seed)
bins = 128 #dx = 1/bins; actual number of columns saved = bins-1, because 1st and last are 0
npots = 2 #ends up being 3*this*(validnth-1)/validnth
validnth = 5 #every nth sample func is saved as validation
sinval = np.sin([[np.pi*i*j/bins for i in range(1,bins)] for j in range(1,bins//2)])
cosval = np.cos([[np.pi*i*j/bins for i in range(1,bins)] for j in range(1,bins//2)])
sqrt2 = np.sqrt(2)

defgrdstate = tf.constant([sqrt2*np.sin(i*np.pi/bins) for i in range(1,bins)])
psi = tf.Variable(defgrdstate)
zerotens = tf.zeros([1])
psil = tf.concat([psi[1:],zerotens],0)
psir = tf.concat([zerotens,psi[:-1]],0)
renorm = tf.assign(psi,tf.divide(psi,tf.sqrt(tf.reduce_mean(tf.square(psi)))))
optimzi = tf.train.GradientDescentOptimizer(0.0625/bins)
reinit = tf.assign(psi,defgrdstate)
init = tf.global_variables_initializer()

potentials = []
validpots = []
wavefuncs = []
validfuncs = []

boxPots = []

#%%      
sess = tf.Session()
sess.run(init)
    
#%%
for i in range(npots):
    boxPots.append(generatepot(127))
 
#%%
for i in range(npots):
    # if i%(npots/100) == 0.0:
    #     print(i/npots * 100, '% complete')
        
    vofx = (boxPots[i])[0]
    
    if max(vofx)!= 0:
        vofx = [vofx[i]/max(vofx) for i in range(len(vofx))]
    
    energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx,1.*bins*bins)),
                                        tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))
    training = optimzi.minimize(energy)
    sess.run(reinit)
    
    for t in range(200000):
        sess.run(training)
        sess.run(renorm)
        
    if i%validnth == 0:
        validpots.append(vofx)
        validfuncs.append(sess.run(psi).tolist())
    else:
        potentials.append(vofx)
        wavefuncs.append(sess.run(psi).tolist())
    
#%%  
with open(path+'test_pots'+str(seed)+'.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(potentials)
with open(path+'valid_pots'+str(seed)+'.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(validpots)
with open(path+'test_out'+str(seed)+'.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(wavefuncs)
with open(path+'valid_out'+str(seed)+'.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(validfuncs)
print ('Output complete')
