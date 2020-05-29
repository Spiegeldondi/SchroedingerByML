import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#%%
path = '/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/'

#%%
bins = 128
seedmax = 20 
trainx = []
trainy = []
validx = []
validy = []

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
def cosShift(re, im):
    shift = np.arctan2(im, re) * 180/np.pi
    shift = np.floor(shift)
    if shift < 0:
        shift += 360
    return shift

#%%
CosInfo = []
# interpolate wavefunction for more precise FFT
for sample in trainy:
    x = np.linspace(0, 126, 127)
    y = sample
    xvals = np.linspace(0, 126, 1000)
    yinterp = np.interp(xvals, x, y)
    
    # spatial resolution and domain
    T = 127
    v = 1/T
    # angular frequency
    w = 2*np.pi*v
    # spatial resolution and domain
    n = 2000
    x = np.linspace(0, T, n)
    xsynth = np.linspace(0, T, n)
    
    # extend and mirror signal for Half Range Fourier Series
    sgnl = np.append(yinterp, -yinterp[::-1])
    
    # frequency domain
    freqs = fftfreq(n) * n
    
    # true physical frequency domain
    realFreqs = freqs > 0
    
    # FFT
    fft_vals = fft(sgnl)
    
    # true physical FFT
    fft_phys = 2.0*np.abs(fft_vals/n)
    
    # compose signal out of sinusoidal paramteres
    sgnl_synth = 0
    Afs = []
    for f in range(1, 17):
        Asynth = fft_phys[f]
        re = fft_vals.real[f]
        im = fft_vals.imag[f]
        s =  cosShift(re, im)*np.pi/180
        sgnl_synth += Asynth * np.cos(f * w * x + s)
        Afs.append(Asynth)
        Afs.append(f)
        Afs.append(s)
    CosInfo.append(Afs)
    
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
W3 = tf.Variable(tf.random_uniform([bins-1, 48], -1./bins, 1./bins))
B3 = tf.Variable(tf.random_uniform([48], -1., 1.))
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
               'Train loss: ', sess.run(costfunc, feed_dict={X: trainx, Y: CosInfo}))#,
               #'Valid loss: ', sess.run(costfunc,feed_dict={X: validx, Y: validy}))
    # actual training 
    sess.run(trainstep, feed_dict={X: trainx, Y: CosInfo})
    
    # store error in lists
    train_loss_list.append(sess.run(costfunc,feed_dict={X: trainx, Y: CosInfo}))
    #valid_loss_list.append(sess.run(costfunc,feed_dict={X: validx, Y: validy}))
    

#%%
# plot analyzed and synthized wavefunctions
plt.plot(sgnl[0:1000], linewidth=6)
plt.plot(sgnl_synth[0:1000], linewidth=6)
plt.tight_layout()

# plot all information
plt.figure(2)
plt.subplot(511)
plt.grid(1)
plt.title('Signal extended as odd function')
plt.plot(sgnl)

plt.subplot(512)
plt.grid(1)
plt.title('Synthized Signal')
plt.plot(xsynth, sgnl_synth)

plt.subplot(513)
plt.grid(1)
plt.title('Real Spectrum')
plt.plot(fft_vals.real, '.')

plt.subplot(514)
plt.grid(1)
plt.title('Imaginary Spectrum')
plt.plot(fft_vals.imag, '.')

plt.subplot(515)
plt.grid(1)
plt.title('Processed Amplitudes (processed absolute values)')
plt.plot(fft_phys, '.')