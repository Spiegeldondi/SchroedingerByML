import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

#%% SINGLE Rectangular potential barrier
path = '/home/domi/Dokumente/RPB_256_1024_float/'

file = path + "descriptors3/test0.csv"
X_train = np.genfromtxt(file, delimiter=',') # load first instance
for i in range(1, 8): # load remaining instances
    file = path + "descriptors3/test"+str(i)+".csv"
    X_train = np.vstack((X_train, np.genfromtxt(file, delimiter=',')))
    
file = path + "descriptors3/test8.csv"
X_valid = np.genfromtxt(file, delimiter=',')
for i in range(9, 10):
    file = path + "descriptors3/test"+str(i)+".csv"
    X_valid = np.vstack((X_valid, np.genfromtxt(file, delimiter=',')))
    
file = path + "targets/test_out0.csv"
y_train = np.genfromtxt(file, delimiter=',')
for i in range(1, 8):
    file = path + "targets/test_out"+str(i)+".csv"
    y_train = np.vstack((y_train, np.genfromtxt(file, delimiter=',')))
    
file = path + "targets/test_out8.csv"
y_valid = np.genfromtxt(file, delimiter=',')
for i in range(9, 10):
    file = path + "targets/test_out"+str(i)+".csv"
    y_valid = np.vstack((y_valid, np.genfromtxt(file, delimiter=',')))

file = path + "features/test_pots0.csv"
potentials = np.genfromtxt(file, delimiter=',')
for i in range(1, 12):
    file = path + "features/test_pots"+str(i)+".csv"
    potentials = np.vstack((potentials, np.genfromtxt(file, delimiter=',')))  


#%% STANDARD APPROACH
file = "/home/domi/Dokumente/SchroedingerByML/potentials/A1/test_pots/test_pots0.csv"
X_train = np.genfromtxt(file, delimiter=',') # load first instance
for i in range(1, 10): # load remaining instances
    file = "/home/domi/Dokumente/SchroedingerByML/potentials/A1/test_pots/test_pots"+str(i)+".csv"
    X_train = np.vstack((X_train, np.genfromtxt(file, delimiter=',')))
    
file = "/home/domi/Dokumente/SchroedingerByML/potentials/A1/valid_pots/valid_pots0.csv"
X_valid = np.genfromtxt(file, delimiter=',')
for i in range(1, 10):
    file = "/home/domi/Dokumente/SchroedingerByML/potentials/A1/valid_pots/valid_pots"+str(i)+".csv"
    X_valid = np.vstack((X_valid, np.genfromtxt(file, delimiter=',')))
    
file = "/home/domi/Dokumente/SchroedingerByML/potentials/A1/test_out/test_out0.csv"
y_train = np.genfromtxt(file, delimiter=',')
for i in range(1, 10):
    file = "/home/domi/Dokumente/SchroedingerByML/potentials/A1/test_out/test_out"+str(i)+".csv"
    y_train = np.vstack((y_train, np.genfromtxt(file, delimiter=',')))
    
file = "/home/domi/Dokumente/SchroedingerByML/potentials/A1/valid_out/valid_out0.csv"
y_valid = np.genfromtxt(file, delimiter=',')
for i in range(1, 10):
    file = "/home/domi/Dokumente/SchroedingerByML/potentials/A1/valid_out/valid_out"+str(i)+".csv"
    y_valid = np.vstack((y_valid, np.genfromtxt(file, delimiter=',')))
#%%   Asdjasd
file = "/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/test_pots/test_pots0.csv"
potentials = np.genfromtxt(file, delimiter=',')
for i in range(1, 12):
    file = "/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/test_pots/test_pots"+str(i)+".csv"
    potentials = np.vstack((potentials, np.genfromtxt(file, delimiter=','))) 
    
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

history = model.fit(X_train, y_train, epochs=1000, 
                    validation_data=(X_valid, y_valid))

#%%
import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5), linewidth=2)
plt.grid(1)
plt.gca().set_ylim(0, 0.5)
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
    
# %% Statistics
    
mse_train_min = np.amin(mse_train)
mse_train_max = np.amax(mse_train)
mse_train_median = np.median(mse_train)
mse_train_mean = np.mean(mse_train)

mse_valid_min = np.amin(mse_valid)
mse_valid_max = np.amax(mse_valid)
mse_valid_median = np.median(mse_valid)
mse_valid_mean = np.mean(mse_valid)

# %% best and worst predictions

e_max_train = mse_train.index(max(mse_train))
e_min_train = mse_train.index(min(mse_train))
e_max_valid = mse_valid.index(max(mse_valid))
e_min_valid = mse_valid.index(min(mse_valid))

# %% average train error
count = 0
for i in mse_train:
    count += i
    
av_train_mse = count/len(mse_train)

# %% average valid error
count = 0
for i in mse_valid:
    count += i
    
av_valid_mse = count/len(mse_valid)

#%%
copy_train = copy.deepcopy(mse_train) 
copy_valid = copy.deepcopy(mse_valid) 

copy_train.sort()
copy_valid.sort()

# e_avg_train = mse_train.index(copy_train[2400])
# e_avg_valid = mse_valid.index(copy_valid[600])

#%% PLOT WORST VALID
plt.grid(1)
# potentials = np.vstack(potentials)
plt.plot(potentials[4000:][192]/max(potentials[4000:][192]), 'grey')
plt.plot(y_pred_valid[192]/max(y_pred_valid[192]), label="Prediction") 
plt.plot(y_valid[192]/max(y_valid[192]), label="Target")
plt.legend()

#%% PLOT AVG VALID
plt.grid(1)
plt.plot(potentials[4000:][100]/max(potentials[4000:][100]), 'grey')
plt.plot(y_pred_valid[100]/max(y_pred_valid[100]), label="Prediction") 
plt.plot(y_valid[100]/max(y_valid[100]), label="Target")
plt.legend()

# %% plot random valid sample
i = np.random.randint(0, len(y_valid))
plt.plot(y_pred_valid[i]) 
plt.plot(y_valid[i])

# %% plot random training sample
i = np.random.randint(0, len(y_train))
plt.plot(y_pred_train[i]) 
plt.plot(y_train[i])

#%%   24
PRD_SGNLS = []
ORG_SGNLS = []
synth_sgnl_mse = []
for i in range(558):
    T = 127
    v = 1/T
    # angular frequency
    w = 2*np.pi*v
    # spatial resolution and domain'
    n = 254
    x = np.linspace(0, T, n)
    
    sgnl_synth = 0   
    sgnl_pred = 0 
    
    freq = 1
    
    for k in np.arange(0, 22, 3):
        sgnl_synth += y_valid[i][k+1] * np.cos(y_valid[i][k] * w * x + y_valid[i][k+2])
        # sgnl_pred += y_pred_valid[i][k+1] * np.cos(y_pred_valid[i][k] * w * x + y_pred_valid[i][k+2])
        if y_pred_valid[i][k+2] < np.pi:
            sgnl_pred += y_pred_valid[i][k+1] * (-1) * np.sin(freq * w * x)
        if y_pred_valid[i][k+2] > np.pi: 
            sgnl_pred += y_pred_valid[i][k+1] * np.sin(freq * w * x)
        
        freq += 1
    PRD_SGNLS.append(sgnl_pred)
    ORG_SGNLS.append(sgnl_synth)
    synth_sgnl_mse.append(mse(sgnl_synth, sgnl_pred))
    
count24 = 0
for k in synth_sgnl_mse:
    count24 += k
avMSE24 = count24/len(synth_sgnl_mse)

plt.plot(sgnl_synth[:127])
plt.plot(sgnl_pred[:127])

"""
LOSS bei Fourier Ansatz muss unbedingt umgerechnet werden auf die tatsächlichen
Wavefunctions
"""

#%% AV AND WORST
import copy
error_valid_composed = copy.deepcopy(synth_sgnl_mse) 
error_valid_composed.sort()

av_ind_synth_sgnl_mse = synth_sgnl_mse.index(error_valid_composed[408])

worst_ind_synth_sgnl_mse = synth_sgnl_mse.index(error_valid_composed[557])

count24 = 0
for k in synth_sgnl_mse:
    count24 += k
avMSE24 = count24/len(synth_sgnl_mse)
#%%
plt.figure(1)
plt.grid(1)
plt.title("Validation Error Distribution (558 Instances)")
plt.xlabel("Error [MSE]")
plt.ylabel("Frequency")
plt.xticks(rotation='vertical')
plt.hist(error_valid_composed, bins=50, edgecolor='black')
#%% PLOT AVG VALID
potentialsNP = np.array(potentials)
aprx_listNP = np.array(aprx_list)

plt.grid(1)
plt.plot(potentialsNP[4000:][av_ind_synth_sgnl_mse]/max(potentialsNP[4000:][av_ind_synth_sgnl_mse]), '--', c='grey', label="original potential")
plt.plot(aprx_listNP[4000:][av_ind_synth_sgnl_mse]/max(aprx_listNP[4000:][av_ind_synth_sgnl_mse]), '--', c='black', label="processed potential")
# plt.plot(X_valid[av_ind_synth_sgnl_mse]/max(X_valid[av_ind_synth_sgnl_mse]), 'grey')
plt.plot(PRD_SGNLS[av_ind_synth_sgnl_mse][:127]/max(PRD_SGNLS[av_ind_synth_sgnl_mse][:127]), linewidth=2, label="Prediction") 
plt.plot(ORG_SGNLS[av_ind_synth_sgnl_mse][:127]/max(ORG_SGNLS[av_ind_synth_sgnl_mse][:127]), linewidth=2, label="Target")
plt.legend(loc="upper right")

#%%   16
PRD_SGNLS = []
ORG_SGNLS = []
synth_sgnl_mse = []
for i in range(1200):
    T = 127
    v = 1/T
    # angular frequency
    w = 2*np.pi*v
    # spatial resolution and domain'
    n = 254
    x = np.linspace(0, T, n)
    
    sgnl_synth = 0   
    sgnl_pred = 0 
    
    freq = 1
    
    for k in np.arange(0, 16, 2):
        sgnl_synth += y_valid[i][k] * np.cos(freq * w * x + y_valid[i][k+1])
        # sgnl_pred += y_pred_valid[i][k] * np.cos(freq * w * x + y_pred_valid[i][k+1])
        if y_pred_valid[i][k+1] < np.pi:
            sgnl_pred += y_pred_valid[i][k] * (-1) * np.sin(freq * w * x)
        if y_pred_valid[i][k+1] > np.pi: 
            sgnl_pred += y_pred_valid[i][k] * np.sin(freq * w * x)
        
        freq += 1
    PRD_SGNLS.append(sgnl_pred)
    ORG_SGNLS.append(sgnl_synth)
    synth_sgnl_mse.append(mse(sgnl_synth, sgnl_pred))
 
count16 = 0
for k in synth_sgnl_mse:
    count16 += k
avMSE16 = count16/len(synth_sgnl_mse)

plt.plot(sgnl_synth[:127])
plt.plot(sgnl_pred[:127])
#%% PLOT ERROR DISTRIBUTION HISTOGRAMS
# Liste aus numpy arrays zu Liste aus floats
mse_valid_aslist = [k[0] for k in mse_valid]
mse_train_aslist = [k[0] for k in mse_train]

plt.figure(1)
plt.grid(1)
plt.title("Validation Error Distribution")
plt.xlabel("Error [MSE]")
plt.ylabel("Frequency")
plt.xticks(rotation='vertical')
plt.hist(mse_valid_aslist, bins=50, edgecolor='black')

plt.figure(2)
plt.grid(1)
plt.title("Training Error Distribution")
plt.xlabel("Error [MSE]")
plt.ylabel("Frequency")
plt.xticks(rotation='vertical')
plt.hist(mse_train_aslist, bins=50, edgecolor='black')

#%%
import csv
with open('/home/domi/Schreibtisch/MailMichele/allTogether/valid_mse.csv', 'w') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(synth_sgnl_mse)
    #%%
with open('/home/domi/Schreibtisch/MailMichele/four24/mse_train_aslist.csv', 'w') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(mse_train_aslist)
    
# %% error by classes
    
mse_train_step = []
mse_valid_step = []

for i in range(len(X_train[0::3])):
    mse_train_step.append(mse(y_train[0::3][i], y_pred_train[0::3][i], multioutput='raw_values'))
for i in range(len(X_valid[0::3])):
    mse_valid_step.append(mse(y_valid[0::3][i], y_pred_valid[0::3][i], multioutput='raw_values'))
    
# %% Statistics
    
mse_train_min_step = np.amin(mse_train_step)
mse_train_max_step = np.amax(mse_train_step)
mse_train_median_step = np.median(mse_train_step)
mse_train_mean_step = np.mean(mse_train_step)

mse_valid_min_step = np.amin(mse_valid_step)
mse_valid_max_step = np.amax(mse_valid_step)
mse_valid_median_step = np.median(mse_valid_step)
mse_valid_mean_step = np.mean(mse_valid_step)
    
    