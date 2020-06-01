import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
#%%
file = "/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/test_pots/test_pots0.csv"
X_train = np.genfromtxt(file, delimiter=',') # load first instance
for i in range(1, 10): # load remaining instances
    file = "/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/test_pots/test_pots"+str(i)+".csv"
    X_train = np.vstack((X_train, np.genfromtxt(file, delimiter=',')))
    
file = "/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/valid_pots/valid_pots0.csv"
X_valid = np.genfromtxt(file, delimiter=',')
for i in range(1, 10):
    file = "/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/valid_pots/valid_pots"+str(i)+".csv"
    X_valid = np.vstack((X_valid, np.genfromtxt(file, delimiter=',')))
    
file = "/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/test_out/test_out0.csv"
y_train = np.genfromtxt(file, delimiter=',')
for i in range(1, 10):
    file = "/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/test_out/test_out"+str(i)+".csv"
    y_train = np.vstack((y_train, np.genfromtxt(file, delimiter=',')))
    
file = "/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/valid_out/valid_out0.csv"
y_valid = np.genfromtxt(file, delimiter=',')
for i in range(1, 10):
    file = "/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/valid_out/valid_out"+str(i)+".csv"
    y_valid = np.vstack((y_valid, np.genfromtxt(file, delimiter=',')))
    
#%% Scikit Learn's MinMaxScaler screwed up
data_max = np.amax(X_train)
X_train = X_train/data_max
X_valid = X_valid/data_max

#%%
model = keras.models.Sequential()
model.add(keras.layers.Dense(127, activation="softplus", input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(127, activation="softplus"))
model.add(keras.layers.Dense(127, activation="softplus"))

model.compile(loss="mean_squared_error", 
              optimizer="adam")

history = model.fit(X_train, y_train, epochs=500, 
                    validation_data=(X_valid, y_valid))

#%%
import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(1)
plt.gca().set_ylim(0, 0.3)
plt.show()

#%%
from sklearn.metrics import mean_squared_error as mse

y_pred_train = model.predict(X_train)
y_pred_valid = model.predict(X_valid)
# %%
mse_train = []
mse_valid = []

for i in range(len(X_train)):
    mse_train.append(mse(y_train[i], y_pred_train[i], multioutput='raw_values'))
for i in range(len(X_valid)):
    mse_valid.append(mse(y_valid[i], y_pred_valid[i], multioutput='raw_values'))

# %% best and worst predictions

e_max_train = mse_train.index(max(mse_train))
e_min_train = mse_train.index(min(mse_train))
e_max_valid = mse_valid.index(max(mse_valid))
e_min_train = mse_valid.index(min(mse_valid))

# %% average train error
count = 0
for i in mse_train:
    count += i
    
av_train_mse = count/len(mse_train)

# %% plot random valid sample
i = np.random.randint(0, len(y_valid))
plt.plot(y_pred_valid[i]) 
plt.plot(y_valid[i])
    
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
        print (step, 
               'Train loss: ', sess.run(costfunc, feed_dict={X: X_train, Y: y_train}),
               'Valid loss: ', sess.run(costfunc,feed_dict={X: X_valid, Y: y_valid}))
    # actual training 
    sess.run(trainstep, feed_dict={X: X_train, Y: y_train})
    
    # store error in lists
    train_loss_list.append(sess.run(costfunc,feed_dict={X: X_train, Y: y_train}))
    valid_loss_list.append(sess.run(costfunc,feed_dict={X: X_valid, Y: y_valid}))


    
