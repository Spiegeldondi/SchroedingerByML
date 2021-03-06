import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# %%
trainx = []
trainy = []
validx = []
validy = []

bins = 128
seedmax = 20

for i in range(seedmax):
    with open('/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/test_pots/test_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainx.append([float(num) for num in row])
    with open('/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/test_out/test_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainy.append([float(num) for num in row])
    with open('/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/valid_pots/valid_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validx.append([float(num) for num in row])
    with open('/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/valid_out/valid_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validy.append([float(num) for num in row])
            
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
            
#%%
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)
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

#%%
trainx_step = trainx[0::3]
trainy_step = trainy[0::3]
validx_step = validx[0::3]
validy_step = validy[0::3]

trainx_four = trainx[2::3]
trainy_four = trainy[2::3]
validx_four = validx[2::3]
validy_four = validy[2::3]

#%%
train_loss_list = []
valid_loss_list = []

#%%
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#%%
for step in range(10000):
    if step % 150 == 0:
        if ic == gslist[gs]:
            gs = gs + 1
            ic = 1
            sess.run(updatelearnrate)
        else:
            ic = ic + 1
    if step %100 == 0:
        print (step, 'Train loss: ',sess.run(costfunc,feed_dict={X: trainx_step, Y: trainy_step}), 'Valid loss: ',sess.run(costfunc,feed_dict={X: validx_four, Y: validy_four}))
    sess.run(trainstep, feed_dict={X: trainx_step, Y: trainy_step})
    
    train_loss_list.append(sess.run(costfunc,feed_dict={X: trainx_step, Y: trainy_step}))
    valid_loss_list.append(sess.run(costfunc,feed_dict={X: validx_four, Y: validy_four}))
    
#%%
plt.grid(1)
plt.title('training and validation error\n(training on step function, validation on fourier series)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot([x for x in range(len(train_loss_list))][0::50], train_loss_list[0::50], 'b', label='train loss')
plt.plot([x for x in range(len(valid_loss_list))][0::50], valid_loss_list[0::50], 'g', label='validation loss')
#plt.plot(valid_loss_list.index(min(valid_loss_list))*10, min(valid_loss_list), 'ro', label='validation loss minimum')
plt.legend()

# %%
plt.grid(1)
#plt.title('Training and Validation Error', fontsize=32)
plt.xlabel('step', fontsize=32)
plt.ylabel('loss', fontsize=32)

plt.plot([x for x in range(len(train_loss_list))][0::50], train_loss_list[0::50], 'orange', linewidth=3, label='train loss')
plt.plot([x for x in range(len(valid_loss_list))][0::50], valid_loss_list[0::50], 'r', linewidth=3, label='validation loss')
#plt.plot(valid_loss_list.index(min(valid_loss_list))*10, min(valid_loss_list), 'ro', label='validation loss minimum')

plt.legend(fontsize=22)

#%%
plt.savefig('/home/domi/Dokumente/BScPresentation/lossSeperateWIDE.png', orientation='landscape', transparent=True)

#%%
with open('/home/domi/Dokumente/SchroedingerByML/lossData/B2/train_loss_list.csv', 'w') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(train_loss_list)
    
with open('/home/domi/Dokumente/SchroedingerByML/lossData/B2/valid_loss_list.csv', 'w') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(valid_loss_list)
    
#%%
potenid = 555
plt.title('prediction vs. ground truth')
plt.xlabel('r')
plt.plot([validx_step[potenid][i]/max(validx_step[potenid]) for i in range(bins - 1)], label='potential')
plt.plot(sess.run(L3,feed_dict={X: [validx_step[potenid]]})[0], label='prediction')
plt.plot(validy_step[potenid], label='target')
plt.legend()
plt.show()

#%%
plt.close()

#%% für Präsi

plt.grid(1)
plt.title('Validation Potential', fontsize=32)
plt.xlabel('x', fontsize=32)
#plt.ylabel('V(x)', fontsize=32)

plt.plot(trainx_four[300], c='m', label='V(x)', linewidth=3)
#mp.plot(sess.run(L3,feed_dict={X: [trainx[potenid]]})[0], label='prediction')
#plt.plot([trainy[potenid][i]/max(trainy[potenid]) for i in range(bins - 1)], label='$\Psi$(x)', linewidth=2)
plt.legend(fontsize=22, loc='upper right')

#%% BALBALBALB 1

hill = [0]*127

hill[20:70] = [0.6]*50
plt.axis([0, 127, 0, 1])
plt.grid(1)
plt.ylabel('V(x)', fontsize=32)
plt.xlabel('x', fontsize=32)
plt.plot(hill, linewidth=3, c='c', label='V(x)')
plt.legend(fontsize=22)

#%% BALBALBALB 2

hill = [0]*127

hill[15:45] = [0.8]*30
hill[60:80] = [0.4]*20
hill[100:120] = [0.6] * 20


plt.axis([0, 127, 0, 1])
plt.grid(1)
plt.ylabel('V(x)', fontsize=32)
plt.xlabel('x', fontsize=32)
plt.plot(hill, linewidth=3, c='c', label='V(x)')
plt.legend(fontsize=22)

#%%
for i in trainx_four[300]:
    if i != 0.0:
        print(trainx_four[300].index(i))

#%% BALBALBALB 3
        
plt.grid(1)
plt.ylabel('V(x)', fontsize=32)
plt.xlabel('x', fontsize=32)
plt.plot(trainx_four[300], linewidth=3, c='grey', label='original potential')
hill = [0]*127

hill[2:19] = [0.3]*17
hill[41:65] = [0.4]*(65-40)
hill[80:106] = [0.7]*(105-80)
hill[105:128] = [0.25]*(128-105)
plt.plot(hill, linewidth=3, c='c', label='simplified potential')
plt.legend(fontsize=22, loc='upper left')
#%%
plt.savefig('/home/domi/Dokumente/BScPresentation/boxSimplified.png', orientation='landscape', transparent=True)
