# schroedinger_nn.py
# This reads the potential training data from genpotential.py and then sets up a neural network with 2 hidden layers.
# Additional tools to output visualize and save the network are in other files.
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

bins = 128
seedmax = 20 # opens seed files 0 - 19. Lost too much data due to kernel crashes, so these got broken up
trainx = []
trainy = []
validx = []
validy = []

#%%
path = '/home/domi/Dokumente/SchroedingerByML/potentials/B1_fixed_win_pots/'

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
W1 = tf.Variable(tf.random_uniform([bins-1, bins-1], -1./bins, 1./bins))
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
for step in range(10000):
    if step % 150 == 0:
        if ic == gslist[gs]:
            gs = gs + 1
            ic = 1
            sess.run(updatelearnrate)
        else:
            ic = ic + 1
    if step %100 == 0:
        print (step, 'Train loss: ',sess.run(costfunc,feed_dict={X: trainx, Y: trainy}), 'Valid loss: ',sess.run(costfunc,feed_dict={X: validx, Y: validy}))
    sess.run(trainstep, feed_dict={X: trainx, Y: trainy})
    
    train_loss_list.append(sess.run(costfunc,feed_dict={X: trainx, Y: trainy}))
    valid_loss_list.append(sess.run(costfunc,feed_dict={X: validx, Y: validy}))
    
# %%

###############
# EXPORT LIST #
###############

with open('/home/domi/Dokumente/SchroedingerByML/lossData/train_loss_list.csv', 'w') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(train_loss_list)
    
with open('/home/domi/Dokumente/SchroedingerByML/lossData/valid_loss_list.csv', 'w') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(valid_loss_list)
    
# %%
    
###############
# IMPORT LIST #
###############

with open('/home/domi/Dokumente/SchroedingerByML/lossData/C/train_loss_list.csv') as csvfile:
    rd = csv.reader(csvfile)
    train_loss = list(rd)[0]
    
with open('/home/domi/Dokumente/SchroedingerByML/lossData/C/valid_loss_list.csv') as csvfile:
    rd = csv.reader(csvfile)
    valid_loss = list(rd)[0]
        
train_loss_list = [float(x) for x in train_loss]
valid_loss_list = [float(x) for x in valid_loss]

train_loss_list = train_loss_list[0::10]
valid_loss_list = valid_loss_list[0::10]

# %%
plt.grid(1)
plt.title('training and validation error')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot([x*10 for x in range(len(train_loss_list))], train_loss_list, 'g', label='train loss')
plt.plot([x*10 for x in range(len(valid_loss_list))], valid_loss_list, 'b', label='validation loss')
plt.plot(valid_loss_list.index(min(valid_loss_list))*10, min(valid_loss_list), 'ro', label='validation loss minimum')
plt.legend()

# %% display_nnout.py

# Makes plots of an individual potential (scaled to unit max), the gradient descent (“correct”) ground state,
# and the neural network predicted ground state
# should be added to notebook containing schroedinger_nn.py
import matplotlib.pyplot as mp
potenid = np.random.randint(0,2400)
mp.plot([trainx[potenid][i]/max(trainx[potenid]) for i in range(bins - 1)], label='potential')
mp.plot(sess.run(L3,feed_dict={X: [trainx[potenid]]})[0], label='prediction')
mp.plot(trainy[potenid], label='target')
plt.legend()
mp.show()