import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#%%
path = '/home/domi/Dokumente/BA/data/3withGaussians/'

#%%
def gaussian(position, height, sigma):
    x = np.linspace(-63, 63, 127)
    pot = np.exp(-0.5 * ((63+x-position) / sigma)**2)*height
    return pot, position, height, sigma


#%%
for seed in range(4, 10):
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
    
    potentials = []
    validpots = []
    wavefuncs = []
    validfuncs = []
    
    allInfo = []
    gaussInfo = []
    k = 0
        
    sess = tf.Session()
    sess.run(init)
    for i in range(npots):
        if i%10 == 0:
            print (str((100.*i)/npots) + '% complete')
        for j in range(3):
            
            pos = np.random.randint(32, 95)
            h = np.random.uniform(256, 512)
            s = np.random.uniform(2, 8)
            
            allInfo.append(gaussian(pos, h, s))
            gaussInfo.append(allInfo[k][1:4])
            vofx = (allInfo[k])[0]
            vofx = [np.float64(n) for n in vofx]
            
            k += 1
            
            energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx,1.*bins*bins)),
                                                tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))
            training = optimzi.minimize(energy)
            sess.run(reinit)
            
            for t in range(2000):
                sess.run(training)
                sess.run(renorm)
                            
            potentials.append(vofx)
            wavefuncs.append(sess.run(psi).tolist())

    with open(path+'test_pots/test_pots'+str(seed)+'.csv', 'w') as f:
        fileout = csv.writer(f)
        fileout.writerows(potentials)
    # with open(path+'valid_pots/valid_pots'+str(seed)+'.csv', 'w') as f:
    #     fileout = csv.writer(f)
    #     fileout.writerows(validpots)
    with open(path+'test_out/test_out'+str(seed)+'.csv', 'w') as f:
        fileout = csv.writer(f)
        fileout.writerows(wavefuncs)
    # with open(path+'valid_out/valid_out'+str(seed)+'.csv', 'w') as f:
    #     fileout = csv.writer(f)
    #     fileout.writerows(validfuncs)

    with open(path+'gaussInfo/'+'test'+str(seed)+'.csv', 'w') as f:
        fileout = csv.writer(f)
        fileout.writerows(gaussInfo)
        
    print ('Output complete')