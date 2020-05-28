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
path = '/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/'

# %% This is not a ... pythonic [barf]... way of reading data, but python is stupid about pointers, so deal with it
for i in range(0, 20): #ACHTUNG ORIGINAL VON 0 BIS SEEDMAX WEG!!!
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
W1 = tf.Variable(tf.random_uniform([10, bins-1], -1./bins, 1./bins))
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

with open('/home/domi/Dokumente/train_loss_list_D1b.csv', 'w') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(train_loss_list)
    
with open('/home/domi/Dokumente/valid_loss_list_D1b.csv', 'w') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(valid_loss_list)

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

# %% display_nnout.py

#ACHTUNG ZUR ZEIT ALLES AUF TRAIN-SET

# potenid = np.random.randint(0, 9600)
potenid = 2747
prediction = sess.run(L3,feed_dict={X: [boxInfo[potenid]]})[0]

plt.grid(1)
#plt.xlabel('x', fontsize=32)
#plt.ylabel('$\Psi$(x)', fontsize=32)

plt.plot([validx[potenid][i]/max(trainx[potenid]) for i in range(bins - 1)], linewidth=3, label='potential V(x)', color='grey')
plt.plot([trainy[potenid][i]/max(trainy[potenid]) for i in range(bins - 1)], c='orange', ls='-.', linewidth=3, label='true $\Psi$(x)')
plt.plot([prediction[i]/max(prediction) for i in range(bins - 1)], 'r', linewidth=3, label='predicted $\Psi$(x)')

plt.legend(fontsize=16, loc='upper right')
plt.show()
#%%
plt.savefig('/home/domi/Schreibtisch/2747.png', orientation='landscape', transparent=True)