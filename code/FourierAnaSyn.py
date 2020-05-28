import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

#%%
def cosShift(re, im):
    shift = np.arctan2(im, re) * 180/np.pi
    shift = np.floor(shift)
    if shift < 0:
        shift += 360
    return shift

#%%
index = 322
x = np.linspace(0, 126, 127)
y = trainy[index]
xvals = np.linspace(0, 126, 1000)
yinterp = np.interp(xvals, x, y)

# plt.subplot(211)
# plt.plot(y)
# plt.subplot(212)
# plt.plot(yinterp)

#%%
# period and frequency
T = 1
v = 1/T

# angular frequency
w = 2*np.pi*v

# spatial resolution and domain
n = 2000
x = np.linspace(0, T, n)

xsynth = np.linspace(0, 2000, n)

# parameters of periodic function
A1 = 12
f1 = 0.5
phi1 = 0
d1 = 0

A2 = 6
f2 = 2
phi2 = 0
d2 = 0

# signal components
sig1 = A1 * np.sin(f1 * w * x + phi1) + d1
sig2 = A2 * np.sin(f2 * w * x + phi2) + d2
sig3 = 2 * np.sin(8*w*x)

# signal assambly
sgnl = sig1 + sig2 + sig3
sgnl = np.append(sgnl, -sgnl[::-1])
sgnl = np.append(yinterp, -yinterp[::-1])
# frequency domain
freqs = fftfreq(n) * n

# true physical frequency domain
realFreqs = freqs > 0

# fft
fft_vals = fft(sgnl)

# true physical fft
fft_phys = 2.0*np.abs(fft_vals/n)

#%%
sgnl_synth = 0
for f in range(1, 200):
    Asynth = fft_phys[f]
    re = fft_vals.real[f]
    im = fft_vals.imag[f]
    s =  cosShift(re, im)*np.pi/180
    sgnl_synth += Asynth * np.cos(f * w * x + s)

#%%  
f1 = 1
f2 = 2
f3 = 3
f4 = 4
f5 = 5
f6 = 6
f7 = 7

Asynth1 = fft_phys[f1]
Asynth2 = fft_phys[f2]
Asynth3 = fft_phys[f3]
Asynth4 = fft_phys[f4]
Asynth5 = fft_phys[f5]
Asynth6 = fft_phys[f6]
Asynth7 = fft_phys[f7]

def cosShift(re, im):
    shift = np.arctan2(im, re) * 180/np.pi
    shift = np.floor(shift)
    if shift < 0:
        shift += 360
    return shift

re1 = fft_vals.real[f1]
im1 = fft_vals.imag[f1]

re2 = fft_vals.real[f2]
im2 = fft_vals.imag[f2]

re3 = fft_vals.real[f3]
im3 = fft_vals.imag[f3]

re4 = fft_vals.real[f4]
im4 = fft_vals.imag[f4]

re5 = fft_vals.real[f5]
im5 = fft_vals.imag[f5]

re6 = fft_vals.real[f6]
im6 = fft_vals.imag[f6]

re7 = fft_vals.real[f7]
im7 = fft_vals.imag[f7]

s1 = cosShift(re1, im1)*np.pi/180
s2 = cosShift(re2, im2)*np.pi/180
s3 = cosShift(re3, im3)*np.pi/180
s4 = cosShift(re4, im4)*np.pi/180
s5 = cosShift(re5, im5)*np.pi/180
s6 = cosShift(re6, im6)*np.pi/180
s7 = cosShift(re7, im7)*np.pi/180


#%%
sgnl_synth = Asynth1 * np.cos(f1 * w * x + s1) + Asynth2 * np.cos(f2 * w * x + s2) + Asynth3 * np.cos(f3 * w * x + s3)
sgnl_synth = sgnl_synth + Asynth4 * np.cos(f4 * w * x + s4) + Asynth5 * np.cos(f5 * w * x + s5) + Asynth6 * np.cos(f6 * w * x + s6)
sgnl_synth = sgnl_synth + Asynth7 * np.cos(f7 * w * x + s7)

#%%
plt.plot(sgnl[0:1000])
plt.plot(sgnl_synth[0:1000])

#%%
plt.figure(1)

plt.subplot(511)
plt.grid(1)
#plt.title('original Signal')
plt.plot(sgnl)

plt.subplot(512)
plt.grid(1)
#plt.title('double inverted Signal')
#plt.plot(sgnl)
plt.plot(xsynth, sgnl_synth)

plt.subplot(513)
plt.grid(1)
#plt.title('Real Part')
plt.plot(fft_vals.real, '.')

plt.subplot(514)
plt.grid(1)
#plt.title('Imag Part')
plt.plot(fft_vals.imag, '.')

plt.subplot(515)
plt.grid(1)
#plt.title('Absolute Value')
plt.plot(fft_phys, '.')