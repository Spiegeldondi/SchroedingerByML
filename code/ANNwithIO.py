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
path = '/home/domi/Dokumente/SchroedingerByML/potentials/super_rand_win_pots/'

# %% This is not a ... pythonic [barf]... way of reading data, but python is stupid about pointers, so deal with it
for i in range(5): #statt 1 gehört hier seedmax
    with open(path+'test_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainx.append([float(num) for num in row])
    with open(path+'test_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainy.append([float(num) for num in row])
    with open(path+'valid_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validx.append([float(num) for num in row])
    with open(path+'valid_out'+str(i)+'.csv', 'r') as csvfile:
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
for step in range(100000):
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
    
    if step %10 == 0:
        train_loss_list.append(sess.run(costfunc,feed_dict={X: trainx, Y: trainy}))
        valid_loss_list.append(sess.run(costfunc,feed_dict={X: validx, Y: validy}))
    
# %%
import csv

# %%

###############
# EXPORT LIST #
###############

with open('train_loss_list.csv', 'w') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(train_loss_list)
    
with open('valid_loss_list.csv', 'w') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(valid_loss_list)
    
# %%
    
###############
# IMPORT LIST #
###############

with open('/home/domi/train_loss_list.csv') as csvfile:
    rd = csv.reader(csvfile)
    train_loss = list(rd)[0]
    
with open('/home/domi/valid_loss_list.csv') as csvfile:
    rd = csv.reader(csvfile)
    valid_loss = list(rd)[0]
    
train_losses = [float(x) for x in train_loss]
valid_losses = [float(x) for x in valid_loss]

# %%
    
import matplotlib.pyplot as plt

# %%
plt.grid(1)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(train_losses, label='train loss')
plt.plot(valid_losses, label='validation loss')
plt.legend()

# %% display_nnout.py

# Makes plots of an individual potential (scaled to unit max), the gradient descent (“correct”) ground state,
# and the neural network predicted ground state
# should be added to notebook containing schroedinger_nn.py
import matplotlib.pyplot as mp
potenid = 1200
mp.plot([trainx[potenid][i]/max(trainx[potenid]) for i in range(bins - 1)], label='potential')
mp.plot(sess.run(L3,feed_dict={X: [trainx[potenid]]})[0], label='prediction')
mp.plot(trainy[potenid], label='target')
plt.legend()
mp.show()

# %%

# save_nn.py
# small tool to save the neural network state. append to schroedinger_nn.py notebook.
import csv
with open('W1.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(sess.run(W1).tolist())
with open('W2.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(sess.run(W2).tolist())
with open('W3.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows(sess.run(W3).tolist())
with open('B1.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows([sess.run(B1).tolist()])
with open('B2.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows([sess.run(B2).tolist()])
with open('B3.csv', 'w') as f:
    fileout = csv.writer(f)
    fileout.writerows([sess.run(B3).tolist()])

# %%

# visualize_nn.py
# outputs bitmaps of the weights and biases from schroedinger_nn.py.
# Sorts them using a Gaussian kernel to increase spatial correlation between weights and nodes.
# Doubles the size of the bitmap before outputting.
# Append to schroedinger_nn.py notebook

def doubler(aray):
    dbled = np.zeros([2*i for i in aray.shape])
    if len(aray.shape) == 1:
        for i in range(aray.shape[0]):
            dbled[2*i] = aray[i]
            dbled[2*i+1] = aray[i]
    elif len(aray.shape) == 2:
        for i in range(aray.shape[0]):
            for j in range(aray.shape[1]):
                dbled[2*i][2*j] = aray[i][j]
                dbled[2*i+1][2*j] = aray[i][j]
                dbled[2*i][2*j+1] = aray[i][j]
                dbled[2*i+1][2*j+1] = aray[i][j]
    return dbled

from PIL import Image
we1 = sess.run(W1)
bi1 = sess.run(B1)
we2 = sess.run(W2)
bi2 = sess.run(B2)
we3 = sess.run(W3)
bi3 = sess.run(B3)

gauswid = 1.
weiscale = []
for i in range(bins-1):
    line = np.exp([-np.square(float(i-j)/gauswid)/2. for j in range(bins-1)])
    line = np.divide(line,sum(line))
    weiscale.append(line.tolist())

weconv1 = np.matmul(we1,weiscale)
weconv2 = np.matmul(we2,weiscale)

sign = 1
mask = np.zeros(bins-1)
for i in range(bins-1):
    ind = (bins-2)/2+int(np.floor((i+1)/2))*sign
    sign = -sign
    mxin = np.argmax(np.add(weconv1[ind],mask))
    swapper = np.identity(bins-1)
    swapper[ind][ind] = 0
    swapper[mxin][mxin] = 0
    swapper[ind][mxin] = 1
    swapper[mxin][ind] = 1
    we1 = np.matmul(we1,swapper)
    weconv1 = np.matmul(weconv1,swapper)
    bi1 = np.matmul(bi1,swapper)
    we2 = np.matmul(swapper,we2)
    mask[ind] = -1.E12

sign = 1
mask = np.zeros(bins-1)
for i in range(bins-1):
    ind = (bins-2)/2+int(np.floor((i+1)/2))*sign
    sign = -sign
    mxin = np.argmax(np.add(weconv2[ind],mask))
    swapper = np.identity(bins-1)
    swapper[ind][ind] = 0
    swapper[mxin][mxin] = 0
    swapper[ind][mxin] = 1
    swapper[mxin][ind] = 1
    we2 = np.matmul(we2,swapper)
    weconv2 = np.matmul(weconv2,swapper)
    bi2 = np.matmul(bi2,swapper)
    we3 = np.matmul(swapper,we3)
    mask[ind] = -1.E12


max1 = max(max(we1.tolist()))
min1 = min(min(we1.tolist()))
wedb1 = doubler(we1)
weight1 = np.divide(np.subtract(wedb1,min1),max1-min1)
wim1 = Image.fromarray((weight1*255).astype(np.uint8),'L')
wim1.save('W1.bmp')
max1 = max(bi1.tolist())
min1 = min(bi1.tolist())
bidb1 = doubler(bi1)
bia1 = np.divide(np.subtract(bidb1,min1),max1-min1)
bias1 = np.array([bia1.tolist() for i in range(32)])
bim1 = Image.fromarray((bias1*255).astype(np.uint8),'L')
bim1.save('B1.bmp')

max2 = max(max(we2.tolist()))
min2 = min(min(we2.tolist()))
wedb2 = doubler(we2)
weight2 = np.divide(np.subtract(wedb2,min2),max2-min2)
wim2 = Image.fromarray((weight2*255).astype(np.uint8),'L')
wim2.save('W2.bmp')
max2 = max(bi2.tolist())
min2 = min(bi2.tolist())
bidb2 = doubler(bi2)
bia2 = np.divide(np.subtract(bidb2,min2),max2-min2)
bias2 = np.array([bia2.tolist() for i in range(32)])
bim2 = Image.fromarray((bias2*255).astype(np.uint8),'L')
bim2.save('B2.bmp')

max3 = max(max(we3.tolist()))
min3 = min(min(we3.tolist()))
wedb3 = doubler(we3)
weight3 = np.divide(np.subtract(wedb3,min3),max3-min3)
wim3 = Image.fromarray((weight3*255).astype(np.uint8),'L')
wim3.save('W3.bmp')
max3 = max(bi3.tolist())
min3 = min(bi3.tolist())
bidb3 = doubler(bi3)
bia3 = np.divide(np.subtract(bidb3,min3),max3-min3)
bias3 = np.array([bia3.tolist() for i in range(32)])
bim3 = Image.fromarray((bias3*255).astype(np.uint8),'L')
bim3.save('B3.bmp')
