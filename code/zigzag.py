import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from scipy.signal import savgol_filter
tf.disable_v2_behavior()

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
potentials = []
validpots = []
wavefuncs = []
validfuncs = []

#%%
sess = tf.Session()
sess.run(init)

#%%
vofx = generatepot(0,1)

#%%
vofx_smooth_np = (np.array(savgol_filter(vofx, 25, 3))).astype('float32')
# vofx_smooth_np = (np.array(vofx)).astype('float32')

vofx_smooth = []

for i in vofx_smooth_np:
    vofx_smooth.append(i)
    
#%%
minmaxlist_smooth = []
indexlist_smooth = []

for i in [0,20,40,60,80,100,120]:
    minmaxlist_smooth.append([min(vofx_smooth[i:i+20]), 
                              vofx_smooth.index(min(vofx_smooth[i:i+20]))])
    minmaxlist_smooth.append([max(vofx_smooth[i:i+20]), 
                              vofx_smooth.index(max(vofx_smooth[i:i+20]))])
    
from operator import itemgetter
minmaxlist_smooth = sorted(minmaxlist_smooth, key=itemgetter(1)) 

domain = [0]
potential = [vofx_smooth[0]] # FÜR ANDERE POTS MUSS DAS MIN AUCH NOCH REIN !!!!!

for i in range(len(minmaxlist_smooth)):
    potential.append((minmaxlist_smooth[i])[0])
    domain.append((minmaxlist_smooth[i])[1])

plt.xlabel('x') 
plt.plot(domain, potential,'o')
plt.plot(vofx_smooth, label='V(x) smooth')

steps = domain[-1] - domain[0]

xvals = np.linspace(domain[0], domain[-1], steps+1)
yinterp = np.interp(xvals, domain, potential)
plt.plot(domain, potential, 'o', label='local extrema')
plt.plot(xvals, yinterp, label='linear interpolation')
plt.legend(loc=1)
#%%
vofx_zz = []

for i in yinterp:
    vofx_zz.append(i)

#%%
'''
LITTLE BIT FISHY TO FILL UP VOFX_ZZ TO 127
'''
# vofx_zz.append(vofx_zz[-1])
# vofx_zz.append(vofx_zz[-1])

#%%
energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx_zz,1.*bins*bins)),
                                    tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))
training = optimzi.minimize(energy)
sess.run(reinit)

#%%
for t in range(20000):
    sess.run(training)
    sess.run(renorm)
    if t%2000 == 0:
        print(t/200, "% complete")
  
#%%
potentials.append(vofx_zz)
wavefuncs.append(sess.run(psi).tolist())
sess.close()

#%% NORMALIZATION
vofx_zz_norm = [v / max(vofx_zz) for v in vofx_zz]
wavefuncs_zz_norm = [psi / max(wavefuncs[0]) for psi in wavefuncs[0]]

#%% PLOT POTENTIAL AND WAVEFUNCTION
plt.xlabel('x')
plt.plot(vofx_zz_norm, label = 'V(x)')
plt.plot(wavefuncs_zz_norm, label = 'Psi(x)')
plt.legend()

#%%
sess = tf.Session()
sess.run(init)

#%%
energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx,1.*bins*bins)),
                                    tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))
training = optimzi.minimize(energy)
sess.run(reinit)

#%%
for t in range(20000):
    sess.run(training)
    sess.run(renorm)
    if t%2000 == 0:
        print(t/200, "% complete")
  
#%%
potentials.append(vofx)
wavefuncs.append(sess.run(psi).tolist())
sess.close()

#%% NORMALIZATION
vofx_norm = [v / max(vofx) for v in vofx]
wavefuncs_norm = [psi / max(wavefuncs[1]) for psi in wavefuncs[1]]

#%% PLOT POTENTIAL AND WAVEFUNCTION
plt.plot(vofx_norm, label='V(x) original')
plt.plot(wavefuncs_norm, label='Psi(x) original')
plt.legend()
#%%
sess = tf.Session()
sess.run(init)

#%%
energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx_smooth,1.*bins*bins)),
                                    tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))
training = optimzi.minimize(energy)
sess.run(reinit)

#%%
for t in range(20000):
    sess.run(training)
    sess.run(renorm)
    if t%2000 == 0:
        print(t/200, "% complete")
  
#%%
potentials.append(vofx_smooth)
wavefuncs.append(sess.run(psi).tolist())
sess.close()

#%% NORMALIZATION
vofx_smooth_norm = [v / max(vofx_smooth) for v in vofx_smooth]
wavefuncs_smooth_norm = [psi / max(wavefuncs[2]) for psi in wavefuncs[2]]

#%% PLOT POTENTIAL AND WAVEFUNCTION
plt.plot(vofx_smooth_norm, label='V(x) smooth')
plt.plot(wavefuncs_smooth_norm, label='Psi(x) smooth')
plt.legend()