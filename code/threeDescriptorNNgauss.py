# schroedinger_nn.py
# This reads the potential training data from genpotential.py and then sets up a neural network with 2 hidden layers.
# Additional tools to output visualize and save the network are in other files.
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
path = '/home/domi/Dokumente/SchroedingerByML/potentials/Dg/'

# %% This is not a ... pythonic [barf]... way of reading data, but python is stupid about pointers, so deal with it
for i in range(0,13): #ACHTUNG ORIGINAL VON 0 BIS SEEDMAX WEG!!!
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
            
#%%
boxInfo = []   
#%%         
for i in range(0,13):
    with open(path+'gaussInfo/'+'test'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            boxInfo.append([float(num) for num in row])           
# %% Normalize Input
            
for k in range(len(trainx)):
    if max(trainx[k])!=0:
            trainx[k] = [trainx[k][i]/max(trainx[k]) for i in range(bins - 1)]

for k in range(len(trainy)):
    if max(trainy[k])!=0:
            trainy[k] = [trainy[k][i]/max(trainy[k]) for i in range(bins - 1)]
            
for k in range(len(validx)):
    if max(validx[k])!=0:
            validx[k] = [validx[k][i]/max(validx[k]) for i in range(bins - 1)]
            
for k in range(len(validy)):
    if max(validy[k])!=0:
            validy[k] = [validy[k][i]/max(validy[k]) for i in range(bins - 1)]

# %%
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)
#have a decaying learning rate so that convergence is faster at first and the fit is better at the end.
#However, by trial and error, the simple exponential decay doesn't work well.
#Trying a method by which the decay happens at hand-specified intervals
startrate = 0.125
gs = 0
gslist = [1,1,2,3,10,20,40,100,200,10000]
ic = 0
learnrate = tf.Variable(startrate, trainable=False)
updatelearnrate = tf.assign(learnrate,tf.multiply(learnrate,0.75))

# %% set up neural network layers. There are shorter ways to do it, but this exposes the guts.
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
#1st hidden layer
W1 = tf.Variable(tf.random_uniform([3, bins-1], -1./bins, 1./bins))
B1 = tf.Variable(tf.random_uniform([bins-1], -1., 1.))
L1 = tf.nn.softplus(tf.matmul(X, W1) + B1)
#2nd hidden layer
W2 = tf.Variable(tf.random_uniform([bins-1, bins-1], -1./bins, 1./bins))
B2 = tf.Variable(tf.random_uniform([bins-1], -1., 1.))
L2 = tf.nn.softplus(tf.matmul(L1, W2) + B2)
#Output layer
W3 = tf.Variable(tf.random_uniform([bins-1, bins-1], -1./bins, 1./bins))
B3 = tf.Variable(tf.random_uniform([bins-1], -1., 1.))
L3 = tf.nn.softplus(tf.matmul(L2, W3) + B3)
#Cost function
costfunc = tf.reduce_mean(tf.square(tf.subtract(L3,Y)))
optimizer = tf.train.GradientDescentOptimizer(learnrate)
trainstep = optimizer.minimize(costfunc)

# %% 
train_loss_list = []
valid_loss_list = []

# %% initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# %%
for step in range(100000):
    if step % 150 == 0:
        if ic == gslist[gs]:
            gs = gs + 1
            ic = 1
            sess.run(updatelearnrate)
        else:
            ic = ic + 1
    if step %100 == 0:
        print (step, 'Train loss: ',sess.run(costfunc,feed_dict={X: boxInfo, Y: trainy}))#, 'Valid loss: ',sess.run(costfunc,feed_dict={X: validx, Y: validy}))
    sess.run(trainstep, feed_dict={X: boxInfo, Y: trainy})
    
    # store error in lists
    train_loss_list.append(sess.run(costfunc,feed_dict={X: boxInfo, Y: trainy}))
    #valid_loss_list.append(sess.run(costfunc,feed_dict={X: validx, Y: validy}))
    
# %%

###############
# EXPORT LIST #
###############

with open('/home/domi/Dokumente/BScPresentation/train_loss_list_A.csv', 'w') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(train_loss_list)
    
with open('/home/domi/Dokumente/BScPresentation/valid_loss_list_A.csv', 'w') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(valid_loss_list)
    
# %%
    
###############
# IMPORT LIST #
###############

with open('/home/domi/Dokumente/SchroedingerByML/lossData/C_winPotFix/train_loss_list.csv') as csvfile:
    rd = csv.reader(csvfile)
    train_loss = list(rd)[0]
    
with open('/home/domi/Dokumente/SchroedingerByML/lossData/C_winPotFix/valid_loss_list.csv') as csvfile:
    rd = csv.reader(csvfile)
    valid_loss = list(rd)[0]
        
train_loss_list = [float(x) for x in train_loss]
valid_loss_list = [float(x) for x in valid_loss]

# train_loss_list = train_loss_list[0::10]
# valid_loss_list = valid_loss_list[0::10]

# %%

########################
# PLOT TRAIN LOSS LIST #
########################

plt.grid(1)
#plt.title('Training and Validation Error', fontsize=32)
plt.xlabel('step', fontsize=32)
plt.ylabel('loss', fontsize=32)

plt.plot([x for x in range(len(train_loss_list))][0::100], train_loss_list[0::100], 'x-', linewidth=3, label='train loss')
#plt.plot([x for x in range(len(valid_loss_list))][0::100], valid_loss_list[0::100], 'r', linewidth=3, label='validation loss')
#plt.plot(valid_loss_list.index(min(valid_loss_list))*10, min(valid_loss_list), 'ro', label='validation loss minimum')

plt.legend(fontsize=22)

#%%
potenid = 0 # np.random.randint(0,2400)

plt.title('Potential V(x)', fontsize=32)
plt.grid(1)
plt.xlabel('x', fontsize=32)
plt.ylabel('V(x)', fontsize=32, color='c')

plt.plot([trainx[potenid][i]/max(trainx[potenid]) for i in range(bins - 1)], 'c', label='V(x)', linewidth=2)
#mp.plot(sess.run(L3,feed_dict={X: [trainx[potenid]]})[0], label='prediction')
#plt.plot([trainy[potenid][i]/max(trainy[potenid]) for i in range(bins - 1)], label='$\Psi$(x)', linewidth=2)
plt.legend(fontsize=18, loc='upper right')



# %% display_nnout.py

#ACHTUNG ZUR ZEIT ALLES AUF TRAIN-SET

potenid = np.random.randint(0,600)
#potenid = 121
prediction = sess.run(L3,feed_dict={X: [boxInfo[potenid]]})[0]

plt.grid(1)
#plt.xlabel('x', fontsize=32)
#plt.ylabel('$\Psi$(x)', fontsize=32)

plt.plot([trainx[potenid][i]/max(trainx[potenid]) for i in range(bins - 1)], linewidth=3, label='potential V(x)', color='grey')
plt.plot([trainy[potenid][i]/max(trainy[potenid]) for i in range(bins - 1)], c='orange', ls='-.', linewidth=3, label='true $\Psi$(x)')
plt.plot([prediction[i]/max(prediction) for i in range(bins - 1)], 'r', linewidth=3, label='predicted $\Psi$(x)')

plt.legend(fontsize=16, loc='upper right')
plt.show()


#%%
count = 0
boxInfo2 = []
for k in range(len(boxInfo)):
    height = boxInfo[k][2]
    width = boxInfo[k][1]-boxInfo[k][0]
    pos = boxInfo[k][0]+(boxInfo[k][1]-boxInfo[k][0])/2
    
    boxInfo2.append([pos, width, height])

    
#%%
'''
INVESTIGATE SOME POTENTIALS
'''
seed = 0
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

potentials = []
validpots = []
wavefuncs = []
validfuncs = []

allInfo = []
boxInfo = []
k = 0
    
sess = tf.Session()
sess.run(init)
for i in range(npots):
    if i%10 == 0:
        print (str((100.*i)/npots) + '% complete')
    for j in range(1):
        
        vofx = trainx[466]
        vofx = [np.float64(n) for n in vofx]
        
        k += 1
        
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
        
#%%
plt.plot(potentials[0])
#%%
plt.plot(wavefuncs[0])
















