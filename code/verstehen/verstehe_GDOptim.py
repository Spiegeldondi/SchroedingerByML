import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#%%
func = np.sin([np.sin(i*np.pi/8) for i in range(1,8)])

psi = tf.Variable(func)

energy = tf.reduce_mean(psi)

optimzi = tf.train.GradientDescentOptimizer(0.0625/8)
training = optimzi.minimize(energy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#%%
for i in range(1000):
    plt.plot(sess.run(psi))
    # print(sess.run(energy))
    sess.run(training)

#%%
sess.close()