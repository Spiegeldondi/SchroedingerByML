import csv
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#%%
in1 = tf.placeholder(tf.float32)
in2 = tf.placeholder(tf.float32)
out = tf.multiply(in1, in2)

sol = []

with tf.Session() as sess:
    sol.append(sess.run(out, feed_dict={in1: [[7.], [8.]], in2: [[2.], [3.]]}))
    
sess.close() 

print(sol)