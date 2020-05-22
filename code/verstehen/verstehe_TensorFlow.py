'''
EINE KLEINE SPIELEREI MIT TENSORFLOW
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#%% tf.random_uniform
tensor = tf.random_uniform([1000], -1./128, 1./128)

with tf.Session() as sess:
    result = tensor.eval()
        
plt.plot(result, 'bo')

#%% tf.subtract

a = tf.constant([[4,5,6], [7,8,9]])
b = tf.constant([[1,2,3], [7,8,9]])
subtraction = tf.subtract(a,b)

with tf.Session() as sess:
    sub = subtraction.eval()
        
sub2np = np.array(sub)
print(sub2np)

#%% tf.multiply --> KEINE MatMul sondern elementweise

a = tf.constant([[4,5,6], [7,8,9]])
b = tf.constant([[1,2,3], [7,8,9]])
multiplication = tf.multiply(a,b)

with tf.Session() as sess:
    mult = multiplication.eval()
    
mult2np = np.array(mult)
print(mult2np)

#%% tf.concat

# bins = 128
# sqrt2 = np.sqrt(2)
# defgrdstate = tf.constant([sqrt2*np.sin(i*np.pi/bins) for i in range(1,bins)]) # length bins-1
# psi = tf.Variable(defgrdstate) # man müsste hier zumindest keiner Variable definieren
# zerotens = tf.zeros([1]) # [0.]
# psil = tf.concat([psi[1:],zerotens],0) # length = bins-1 (1. Element geskipped aber 0 angehängt)
# psir = tf.concat([zerotens,psi[:-1]],0) # length = bins-1 (1. Element geskipped aber 0 angehängt)
# renorm = tf.assign(psi,tf.divide(psi,tf.sqrt(tf.reduce_mean(tf.square(psi))))) # normiert psi

# i = 0.02
# j = 0
# npots = 1


# vofx = generatepot(j,(1.*i)/npots)

# energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx,1.*bins*bins)),
#                                                 tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))

# addition = tf.add(psil,psir)
# asdf = tf.multiply(addition,psi)

# init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     init.run()
#     grdstate1 = defgrdstate.eval()
#     psi_l = psil.eval()
#     psi_r = psir.eval()
#     grdstate2 = psi.eval()
#     rn = renorm.eval()
#     en = energy.eval()
#     add = addition.eval()
#     fdsa = asdf.eval()

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
bins = 16
sqrt2 = np.sqrt(2)

defgrdstate = tf.constant([sqrt2*np.sin(i*np.pi/bins) for i in range(1,bins)])

psi = tf.Variable(defgrdstate)

zerotens = tf.zeros([1])

psil = tf.concat([psi[1:],zerotens],0)
psir = tf.concat([zerotens,psi[:-1]],0)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    defgrdstateX = defgrdstate.eval()
    psiX = psi.eval()
    psilX = psil.eval()
    psirX = psir.eval()

#%%
plt.plot(defgrdstateX)

#%%
plt.plot(psiX)
print(psiX)

#%%
plt.plot(psilX)
print(psilX)

#%%
plt.plot(psirX)
print(psirX)

#%%
renorm = tf.assign(psi,tf.divide(psi,tf.sqrt(tf.reduce_mean(tf.square(psi)))))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    renormX = renorm.eval()
    
#%%  
plt.plot(renormX)
print(renormX)

#%%
optimzi = tf.train.GradientDescentOptimizer(0.0625/bins)
reinit = tf.assign(psi,defgrdstate)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    reinitX = reinit.eval()

#%%  
plt.plot(reinitX)
print(reinitX)

#%%
j = 0
i = 0.02
npots = 1

vofx = generatepot(j,(1.*i)/npots)
energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx,1.*bins*bins)),
                                         tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))

with tf.Session() as sess:
    init.run()
    energyX = energy.eval()

#%%
print(energyX)










