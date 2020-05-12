import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#%%
def generatepot1(length):
    pot = [0]*length    
    check = False
    
    while check == False:
        posM = np.random.randint(0, len(pot)+1)
        w = np.random.randint(0, len(pot)+1)
        h = np.random.randint(100, 300)
        if w%2 != 0 and ((posM+int(w/2)) < len(pot)) and ((posM-int(w/2)) >= 0):
            a = posM - int(w/2)
            b = posM + 1 + int(w/2)
            pot[a:b] = [h]*w
            check = True
    
    return pot, a, (b-1), h

#%%
def subexp(expon):
    return np.power(abs(np.log(np.random.uniform())),expon)

def generatepot(style,param): #0=step,1=linear,2=fourier; 0-1 "jaggedness" scale
    mu = 1. + bins*param #mean number of jump points for styles 0 + 1
    forxp = 2.5 - 2*param #fourier exponent for style 2
    scale = 5.0*(np.pi*np.pi*0.5) # energy scale
    if style < 2:
        dx = bins/mu
        xlist = [-dx/2]
        while xlist[-1] < bins:
            xlist.append(xlist[-1]+dx*subexp(1.))
        vlist = [scale*subexp(2.) for k in range(len(xlist))]
        k = 0
        poten = []
        for l in range(1,bins):
            while xlist[k+1] < l:
                k = k + 1
            if style == 0:
                poten.append(vlist[k])
            else:
                poten.append(vlist[k]+(vlist[k+1]-vlist[k])*(l-xlist[k])/(xlist[k+1]-xlist[k]))
    else:
        sincoef = [(2*np.random.randint(2)-1.)*scale*subexp(2.)/np.power(k,forxp) for k in range(1,bins//2)]
        coscoef = [(2*np.random.randint(2)-1.)*scale*subexp(2.)/np.power(k,forxp) for k in range(1,bins//2)]
        zercoef = scale*subexp(2.)
        poten = np.maximum(np.add(np.add(np.matmul(sincoef,sinval),np.matmul(coscoef,cosval)),zercoef),0).tolist()
    return poten

#%%
seed = 1
np.random.seed(seed)
bins = 128 #dx = 1/bins; actual number of columns saved = bins-1, because 1st and last are 0
npots = 1 #ends up being 3*this*(validnth-1)/validnth
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
potentials = []
validpots = []
wavefuncs = []
validfuncs = []
#%%
sess = tf.Session()
sess.run(init)
#%%
vofx = generatepot(0, 0.02)

energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx,1.*bins*bins)),
                                    tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))
training = optimzi.minimize(energy)
sess.run(reinit)
#%%
for t in range(20000):
    sess.run(training)
    sess.run(renorm)
    
potentials.append(vofx)
wavefuncs.append(sess.run(psi).tolist())
#%%
sess.close()
#%%
sess = tf.Session()
sess.run(init)
#%%
vofx = [k*100 for k in vofx]

energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx,1.*bins*bins)),
                                    tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))
training = optimzi.minimize(energy)
sess.run(reinit)
#%%
for t in range(200):
    sess.run(training)
    sess.run(renorm)
    
potentials.append(vofx)
wavefuncs.append(sess.run(psi).tolist())
#%%
sess.close()  
#%%
plt.figure()
plt.subplot(221)
plt.plot(potentials[0])

plt.subplot(222)
plt.plot(potentials[1])
plt.show()

plt.subplot(223)
plt.plot(wavefuncs[0])

plt.subplot(224)
plt.plot(wavefuncs[1])
plt.show()
  
#%%
plt.plot(potentials[0])
#%%
plt.plot(wavefuncs[0])
