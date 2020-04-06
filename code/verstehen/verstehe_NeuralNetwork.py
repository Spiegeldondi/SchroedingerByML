import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

X_train = []
y_train = []
X_valid = []
y_valid = []

#%% import Data

i = 0

with open('test_pots'+str(i)+'.csv', 'r') as csvfile:
    flurg = csv.reader(csvfile)
    for row in flurg:
        X_train.append([float(num) for num in row])
with open('test_out'+str(i)+'.csv', 'r') as csvfile:
    flurg = csv.reader(csvfile)
    for row in flurg:
        y_train.append([float(num) for num in row])
with open('valid_pots'+str(i)+'.csv', 'r') as csvfile:
    flurg = csv.reader(csvfile)
    for row in flurg:
        X_valid.append([float(num) for num in row])
with open('valid_out'+str(i)+'.csv', 'r') as csvfile:
    flurg = csv.reader(csvfile)
    for row in flurg:
        y_valid.append([float(num) for num in row])

#%% convert list of lists to numpy array and scale input features      
        
X_train = np.array(X_train)/max([max(x) for x in X_train])
y_train = np.array(y_train)/max([max(y) for y in y_train])
X_valid = np.array(X_valid)/max([max(x) for x in X_valid])
y_valid = np.array(y_valid)/max([max(y) for y in y_valid])

#%% set weights and biases

bins = 128

w1 = np.zeros((bins-1,bins-1))
b1 = np.zeros((bins-1,))

w2 = np.zeros((bins-1,bins-1))
b2 = np.zeros((bins-1,))

w3 = np.zeros((bins-1,bins-1))
b3 = np.zeros((bins-1,))

w1 = np.random.uniform(-1./bins, 1./bins, (127, 127))
b1 = np.random.uniform(-1., 1., (127,))
w2 = np.random.uniform(-1./bins, 1./bins, (127, 127))
b2 = np.random.uniform(-1., 1., (127,))
w3 = np.random.uniform(-1./bins, 1./bins, (127, 127))
b3 = np.random.uniform(-1., 1., (127,))
        
#%% create model
        
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=(127,)))
model.add(keras.layers.Dense(127, activation="softplus")) 
model.add(keras.layers.Dense(127, activation="softplus"))
model.add(keras.layers.Dense(127, activation="softplus"))
model.set_weights([w1, b1, w2, b2, w3, b3])

model.summary()

#%% compile model

model.compile(loss="mse", optimizer="sgd", metrics=["accuracy"])

#%% train model

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

#%% generate test potential

def generatepot(style,param): #0=step,1=linear,2=fourier; 0-1 "jaggedness" scale
    mu = 1. + bins*param #mean number of jump points for styles 0 + 1
    forxp = 2.5 - 2*param #fourier exponent for style 2
    scale = 5.0*(np.pi*np.pi*0.5) # energy scale
    if style < 2:
        dx = bins/mu
        xlist = [-dx/2]
        while xlist[-1] < bins:
            xlist.append(xlist[-1]+dx*subexp(1.))
        vlist = [scale*subexp(2.) for k in range(len(xlist))]
        k = 0
        poten = []
        for l in range(1,bins):
            while xlist[k+1] < l:
                k = k + 1
            if style == 0:
                poten.append(vlist[k])
            else:
                poten.append(vlist[k]+(vlist[k+1]-vlist[k])*(l-xlist[k])/(xlist[k+1]-xlist[k]))
    else:
        sincoef = [(2*np.random.randint(2)-1.)*scale*subexp(2.)/np.power(k,forxp) for k in range(1,bins//2)]
        coscoef = [(2*np.random.randint(2)-1.)*scale*subexp(2.)/np.power(k,forxp) for k in range(1,bins//2)]
        zercoef = scale*subexp(2.)
        poten = np.maximum(np.add(np.add(np.matmul(sincoef,sinval),np.matmul(coscoef,cosval)),zercoef),0).tolist()
    return poten

#%% 

def subexp(expon):
    return np.power(abs(np.log(np.random.uniform())),expon)    

bins = 128
X_test = generatepot(0, 0.02)

#%% query model

X_test = np.array(X_test)

y_pred = model.predict(X_train)