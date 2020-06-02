import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

#%% SINGLE Rectangular potential barrier
path = '/home/domi/Dokumente/SchroedingerByML/potentials/betterBoxPots/'

file = path + "boxInfo/test1.csv"
X_train = np.genfromtxt(file, delimiter=',') # load first instance
for i in range(2, 8): # load remaining instances
    file = path + "boxInfo/test"+str(i)+".csv"
    X_train = np.vstack((X_train, np.genfromtxt(file, delimiter=',')))
    
file = path + "test_out/test_out1.csv"
y_train = np.genfromtxt(file, delimiter=',')
for i in range(2, 8):
    file = path + "test_out/test_out"+str(i)+".csv"
    y_train = np.vstack((y_train, np.genfromtxt(file, delimiter=',')))

file = path + "test_pots/test_pots1.csv"
potentials = np.genfromtxt(file, delimiter=',')
for i in range(2, 8):
    file = path + "test_pots/test_pots"+str(i)+".csv"
    potentials = np.vstack((potentials, np.genfromtxt(file, delimiter=',')))  


#%% STANDARD APPROACH
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
    
# %% FOURIER FOURIER FOURIER
y_train = np.array([fourierDecomp(x) for x in y_train])
y_valid = np.array([fourierDecomp(x) for x in y_valid])
    
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

history = model.fit(X_train, y_train, epochs=2000, 
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

# %% plot random training sample
i = np.random.randint(0, len(y_train))
#plt.plot(y_pred_train[i]) 
#plt.plot(y_train[i])

#%%   
T = 127
v = 1/T
# angular frequency
w = 2*np.pi*v
# spatial resolution and domain'
n = 254
x = np.linspace(0, T, n)

sgnl_synth = 0    
   
sgnl_synth += y_train[i][1] * np.cos(y_train[i][0] * w * x + y_train[i][2])
sgnl_synth += y_train[i][4] * np.cos(y_train[i][3] * w * x + y_train[i][5])
sgnl_synth += y_train[i][7] * np.cos(y_train[i][6] * w * x + y_train[i][8])
sgnl_synth += y_train[i][10] * np.cos(y_train[i][9] * w * x + y_train[i][11])
sgnl_synth += y_train[i][13] * np.cos(y_train[i][12] * w * x + y_train[i][14])
sgnl_synth += y_train[i][16] * np.cos(y_train[i][15] * w * x + y_train[i][17])
sgnl_synth += y_train[i][19] * np.cos(y_train[i][18] * w * x + y_train[i][20])
sgnl_synth += y_train[i][22] * np.cos(y_train[i][21] * w * x + y_train[i][23])

plt.plot(sgnl_synth[:127])

sgnl_pred = 0

sgnl_pred += y_pred_train[i][1] * np.cos(y_pred_train[i][0] * w * x + y_pred_train[i][2])
sgnl_pred += y_pred_train[i][4] * np.cos(y_pred_train[i][3] * w * x + y_pred_train[i][5])
sgnl_pred += y_pred_train[i][7] * np.cos(y_pred_train[i][6] * w * x + y_pred_train[i][8])
sgnl_pred += y_pred_train[i][10] * np.cos(y_pred_train[i][9] * w * x + y_pred_train[i][11])
sgnl_pred += y_pred_train[i][13] * np.cos(y_pred_train[i][12] * w * x + y_pred_train[i][14])
sgnl_pred += y_pred_train[i][16] * np.cos(y_pred_train[i][15] * w * x + y_pred_train[i][17])
sgnl_pred += y_pred_train[i][19] * np.cos(y_pred_train[i][18] * w * x + y_pred_train[i][20])
sgnl_pred += y_pred_train[i][22] * np.cos(y_pred_train[i][21] * w * x + y_pred_train[i][23])

plt.plot(sgnl_pred[:127])

"""
LOSS bei Fourier Ansatz muss unbedingt umgerechnet werden auf die tatsächlichen
Wavefunctions
"""