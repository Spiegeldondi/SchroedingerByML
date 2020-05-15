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
path = '/home/domi/Dokumente/SchroedingerByML/potentials/D/'

# %% This is not a ... pythonic [barf]... way of reading data, but python is stupid about pointers, so deal with it
for i in range(0,3): #ACHTUNG ORIGINAL VON 0 BIS SEEDMAX WEG!!!
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
for i in range(0,3):
    with open(path+'boxInfo/'+'test'+str(i)+'.csv', 'r') as csvfile:
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
for step in range(10000):
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
plt.savefig('/home/domi/Dokumente/BScPresentation/lossWindowWIDE.png', orientation='landscape', transparent=True)

#%%
lw = 2

plt.grid(1)
plt.title('Randomly Generated Potentials', fontsize=32)
plt.xlabel('x', fontsize=32)
plt.ylabel('V(x)', fontsize=32)

i_step = 0
plot_step = []
norm_step = max(trainx[i_step])
list_step = [k/norm_step for k in trainx[i_step]]
plt.plot(list_step, 'c', linewidth=lw, label='step')

i_linear = 10
plot_linear = []
norm_linear = max(trainx[i_linear])
list_linear = [k/norm_linear for k in trainx[i_linear]]
plt.plot(list_linear,'m', linewidth=lw, label='linear')

i_fourier = 17
plot_fourier = []
norm_fourier = max(trainx[i_fourier])
list_fourier = [k/norm_fourier for k in trainx[i_fourier]]
plt.plot(list_fourier, 'y', linewidth=lw, label='fourier')

plt.legend(fontsize=18)

#%%
plt.savefig('/home/domi/Dokumente/BScPresentation/ThreePotentials3', orientation='landscape', transparent=True)

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

#%%
# Create some mock data
data1 = [trainx[potenid][i]/max(trainx[potenid]) for i in range(bins - 1)]
data2 = [trainy[potenid][i]/max(trainy[potenid]) for i in range(bins - 1)]

fig, ax1 = plt.subplots()

plt.title('Potential V(x) and Wavefunction $\Psi$(x)', fontsize=32)
plt.grid(1)

color = 'c'
ax1.set_xlabel('x', fontsize=32)
ax1.set_ylabel('V(x)', fontsize=32, color=color)
ax1.plot(data1, linewidth=2, color=color, label='V(x)')
ax1.plot(data2, linewidth=2, color='b', label='$\Psi$(x)')
ax1.tick_params(axis='y', labelcolor=color)
plt.legend(fontsize=18, loc='upper right')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('$\Psi$(x)', fontsize=32, color=color)  # we already handled the x-label with ax1
#ax2.plot(data2, linewidth=2, color=color, label='$\Psi$(x)')
ax2.tick_params(axis='y', labelcolor=color)

#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

#%%
plt.savefig('/home/domi/Dokumente/BScPresentation/PotentialWithWavefunction', orientation='landscape', transparent=True)

# %% display_nnout.py

#ACHTUNG ZUR ZEIT ALLES AUF TRAIN-SET

#potenid = np.random.randint(0,1800)
potenid = 121
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
plt.savefig('/home/domi/Dokumente/BScPresentation/ex5', orientation='landscape', transparent=True)

#%% Histogram
targ_norm = [validy[potenid][i]/max(validy[potenid]) for i in range(bins - 1)]
pred_norm = [prediction[i]/max(prediction) for i in range(bins - 1)]
plt.plot(targ_norm, pred_norm, 'bx')

#%% Plot of Window Pots
for x in range(len(trainx)):
    if x%(1200) == 0:
        k = np.random.randint(1,4800)
        demo = [trainx[k][i]/max(trainx[k]) for i in range(127)]
        plt.grid(1)
        plt.xlabel('x', fontsize=32)
        plt.ylabel('V(x)', fontsize=32)
        plt.plot(demo, linewidth=3)
        
#%%
plt.savefig('/home/domi/Dokumente/BScPresentation/windowPots', orientation='landscape', transparent=True)