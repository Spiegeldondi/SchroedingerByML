import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#%%

bins = 128
seedmax = 20 # opens seed files 0 - 19. Lost too much data due to kernel crashes, so these got broken up
trainx = []
trainy = []
validx = []
validy = []

#%%
path = '/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/'
# path = '/home/domi/Dokumente/SchroedingerByML/potentials/D1/'

# %% This is not a ... pythonic [barf]... way of reading data, but python is stupid about pointers, so deal with it
for i in range(seedmax):
    with open(path+'test_pots/test_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainx.append([float(num) for num in row])
    with open(path+'test_out/test_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainy.append([float(num) for num in row])
    with open(path+'valid_pots/valid_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validx.append([float(num) for num in row])
    with open(path+'valid_out/valid_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validy.append([float(num) for num in row])
            
# %%
import copy
n = np.random.randint(0, 9600)
#n = 9086

test = copy.deepcopy(trainx[n]) 

av = round((max(test)+min(test))/3)
top = max(test)

for k in test:
    if k <= av:
        test[test.index(k)]=0
    if k > av:
        test[test.index(k)]=top


plt.figure(2)
plt.plot(np.array(trainx[n])/top)
plt.plot(np.array(test)/top)
plt.plot(np.array(trainy[n])/max(trainy[n]))

# %%
seed = 42
np.random.seed(seed)
bins = 128 #dx = 1/bins; actual number of columns saved = bins-1, because 1st and last are 0
npots = 200 #ends up being 3*this*(validnth-1)/validnth
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

#%%
i = 4
j = 2

vofx = aprx
# %%
sess = tf.Session()
sess.run(init)

energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx,1.*bins*bins)),
                                    tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))
training = optimzi.minimize(energy)
sess.run(reinit)
        
for t in range(20000):
    sess.run(training)
    sess.run(renorm)

# plt.figure(1)
# plt.plot(vofx)
# plt.figure(2)
# wf = np.array(sess.run(psi))
# plt.plot(wf/np.amax(wf))
           
# sess.close()















































