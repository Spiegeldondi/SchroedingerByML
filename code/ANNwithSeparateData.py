import csv
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#%%
trainx = []
trainy = []
validx = []
validy = []

bins = 128
seedmax = 20

for i in range(seedmax):
    with open('/home/domi/schroedinger/test_pots/test_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainx.append([float(num) for num in row])
    with open('/home/domi/schroedinger/test_out/test_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainy.append([float(num) for num in row])
    with open('/home/domi/schroedinger/valid_pots/valid_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validx.append([float(num) for num in row])
    with open('/home/domi/schroedinger/valid_out/valid_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validy.append([float(num) for num in row])
            
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
for step in range(100000):
    if step % 150 == 0:
        if ic == gslist[gs]:
            gs = gs + 1
            ic = 1
            sess.run(updatelearnrate)
        else:
            ic = ic + 1
    if step %100 == 0:
        train_loss_list.append(sess.run(costfunc,feed_dict={X: trainx_step, Y: trainy_step}))
        valid_loss_list.append(sess.run(costfunc,feed_dict={X: validx_four, Y: validy_four}))
        print (step, 'Train loss: ',sess.run(costfunc,feed_dict={X: trainx_step, Y: trainy_step}), 'Valid loss: ',sess.run(costfunc,feed_dict={X: validx_four, Y: validy_four}))
    sess.run(trainstep, feed_dict={X: trainx_step, Y: trainy_step})
    
#%%
import matplotlib.pyplot as plt
plt.grid(1)
plt.title('training and validation error\n(training on step function, validation on fourier series)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot([x*10 for x in range(len(train_loss_list))], train_loss_list, 'b', label='train loss')
plt.plot([x*10 for x in range(len(valid_loss_list))], valid_loss_list, 'g', label='validation loss')
plt.plot(valid_loss_list.index(min(valid_loss_list))*10, min(valid_loss_list), 'ro', label='validation loss minimum')
plt.legend()

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
plt.plot([validx_four[potenid][i]/max(validx_four[potenid]) for i in range(bins - 1)], label='potential')
plt.plot(sess.run(L3,feed_dict={X: [validx_four[potenid]]})[0], label='prediction')
plt.plot(validy_four[potenid], label='target')
plt.legend()
plt.show()

#%%
plt.close()

























