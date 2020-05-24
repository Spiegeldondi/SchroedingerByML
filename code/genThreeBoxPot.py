import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#%%
path = '/home/domi/Dokumente/SchroedingerByML/potentials/D3/'

#%%
def generatepot(length):
    pot = [0]*length    
    check = False
    
    while check == False:
        posM = np.random.randint(0, len(pot)+1)
        w = np.random.randint(0, len(pot)+1)
        h = np.random.randint(100, 1000) # min and max height of potential
        if w%2 != 0 and ((posM+int(w/2)) < len(pot)) and ((posM-int(w/2)) >= 0):
            a = posM - int(w/2)
            b = posM + 1 + int(w/2)
            pot[a:b] = [h]*w
            check = True
    
    return [pot, [a, (b-1), h]]

#%%
for seed in range(5, 13): 
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
    boxInfo = []
        
    sess = tf.Session()
    sess.run(init)
    for i in range(npots):
        if i%10 == 0:
            print (str((100.*i)/npots) + '% complete')
        for j in range(3):
            
            # generate individual boxes
            a = generatepot(42)
            b = generatepot(42)
            c = generatepot(43)
            
            # shift start/end points of 2. and 3. rect. pot. 
            b[1][0] += 42
            b[1][1] += 42
            c[1][0] += 84
            c[1][1] += 84
            
            # assemble potential
            vofx = []
            vofx.extend(a[0])
            vofx.extend(b[0])
            vofx.extend(c[0])
            vofx = [np.float64(n) for n in vofx]
            
            # store start/end/height of boxes
            placeholder = []
            placeholder.extend(a[1])
            placeholder.extend(b[1])
            placeholder.extend(c[1])
            boxInfo.append(placeholder)
            
            energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi),tf.add(vofx,1.*bins*bins)),
                                                tf.multiply(tf.multiply(tf.add(psil,psir),psi),0.5*bins*bins)))
            training = optimzi.minimize(energy)
            sess.run(reinit)
            
            for t in range(2000):
                sess.run(training)
                sess.run(renorm)
                
            # if i%validnth == 0:
            #     validpots.append(vofx)
            #     validfuncs.append(sess.run(psi).tolist())
            # else:
            #     potentials.append(vofx)
            #     wavefuncs.append(sess.run(psi).tolist())
                
            potentials.append(vofx)
            wavefuncs.append(sess.run(psi).tolist())
  
    with open(path+'test_pots/test_pots'+str(seed)+'.csv', 'w') as f:
        fileout = csv.writer(f)
        fileout.writerows(potentials)
    with open(path+'valid_pots/valid_pots'+str(seed)+'.csv', 'w') as f:
        fileout = csv.writer(f)
        fileout.writerows(validpots)
    with open(path+'test_out/test_out'+str(seed)+'.csv', 'w') as f:
        fileout = csv.writer(f)
        fileout.writerows(wavefuncs)
    with open(path+'valid_out/valid_out'+str(seed)+'.csv', 'w') as f:
        fileout = csv.writer(f)
        fileout.writerows(validfuncs)

    with open(path+'boxInfo/'+'test'+str(seed)+'.csv', 'w') as f:
        fileout = csv.writer(f)
        fileout.writerows(boxInfo)
        
    print ('Output complete')