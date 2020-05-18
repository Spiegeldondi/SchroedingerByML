import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

#%%
path = '/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/'

#%%
bins = 128
seedmax = 20
trainx = []
trainy = []
validx = []
validy = []

#%%
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
            
#%%
model = keras.models.Sequential([
    keras.layers.Dense(),
    keras.layers.Dense(),
    keras.layers.Dense(),  
])

#%%
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model.summary()

#%%
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

#%%
X_train = np.array(trainx)
y_train = np.array(trainy)

model = keras.models.Sequential()
model.add(keras.layers.Dense(127, input_shape=[127,], activation="softplus"))
model.add(keras.layers.Dense(127, activation="softplus"))
model.add(keras.layers.Dense(127, activation="softplus"))

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model.summary()

#%%
model.compile(loss="mean_squared_error",
              optimizer=tf.keras.optimizers.SGD(
                  learning_rate=0.01))

history = model.fit(X_train, y_train, epochs=10000)