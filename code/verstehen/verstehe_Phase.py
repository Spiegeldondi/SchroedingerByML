import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

#%%
# period and frequency
T = 1
v = 1/T

# angular frequency
w = 2*np.pi*v

# spatial resolution and domain
n = 1000
x = np.linspace(0, T, n)

# parameters of periodic function
A1 = 8
f1 = 1
phi1 = np.pi*0.5
d1 = 0

A2 = 6
f2 = 2
phi2 = -1*np.pi*1
d2 = 0

A3 = 0
f3 = 3
phi3 = np.pi*0
d3 = 0

# signal components
sig1 = A1 * np.sin(f1 * w * x + phi1) + d1
sig2 = A2 * np.sin(f2 * w * x + phi2) + d2
sig3 = A3 * np.sin(f3 * w * x + phi3) + d3

# s2 = 0
# s3 = 0

# signal assambly
sgnl = sig1 + sig2 + sig3

# frequency domain
freqs = fftfreq(n) * n

# true physical frequency domain
realFreqs = freqs > 0

# fft
fft_vals = fft(sgnl)

# true physical fft
fft_phys = 2.0*np.abs(fft_vals/n)

#%% für Cosnis. funtioniert für pi aus [0, 2*pi)
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

s1 = cosShift(re1, im1)*np.pi/180
s2 = cosShift(re2, im2)*np.pi/180
s3 = cosShift(re3, im3)*np.pi/180

print(s1,s2,s3)

#%%
sgnl_synth = A1 * np.cos(f1 * w * x + s1) + d1 + A2 * np.cos(f2 * w * x + s2) + d2 + A3 * np.cos(f3 * w * x + s3) + d3

plt.figure(1)

plt.subplot(411)
plt.grid(1)
#plt.title('Signal')
plt.plot(sgnl)
plt.plot(sgnl_synth)

plt.subplot(412)
plt.grid(1)
#plt.title('Real Part')
plt.plot(fft_vals.real, '.')

plt.subplot(413)
plt.grid(1)
#plt.title('Imag Part')
plt.plot(fft_vals.imag, '.')

plt.subplot(414)
plt.grid(1)
#plt.title('Absolute Value')
plt.plot(np.abs(fft_vals), '.')


#%%

'''
Grund für die 0-Punkt-Symmetrie des synthetischen Signals hängt damit zusammen, 
dass auch negative Frequenzen zugelassen sind und das ganze dann irgendwie
für negative und positive frequenzen geplottet wird.
'''

yintercept = round(fft_vals[0].real) / n
phi1 = 0
phi2 = 0
phi3 = 0
# 1. harmonic
b1 = np.where(fft_phys[realFreqs] == np.sort(fft_phys[realFreqs])[::-1][0])[0][0]+1
A1 = (fft_phys[realFreqs][b1-1])
# 2. harmonic
b2 = np.where(fft_phys[realFreqs] == np.sort(fft_phys[realFreqs])[::-1][1])[0][0]+1
A2 = (fft_phys[realFreqs][b2-1])
# 3. harmonic
b3 = np.where(fft_phys[realFreqs] == np.sort(fft_phys[realFreqs])[::-1][2])[0][0]+1
A3 = (fft_phys[realFreqs][b3-1])
# 4. harmonic
b4 = np.where(fft_phys[realFreqs] == np.sort(fft_phys[realFreqs])[::-1][3])[0][0]+1 # artefact due to poor resolution !!
A4 = (fft_phys[realFreqs][b4-1])

sgnl_synth = yintercept + A1 * np.sin(b1 * w * x + phi1) + sig1# + A3 * np.sin(b3 * w * x + phi3) + A4 * np.sin(b4 * w * x + phi3) )
# sgnl_synth = yintercept + ( A1 * np.sin(b1 * w * x + phi1))

# FALLS ALSO die unterste schwingung eine Sinus schwingung mit Fq 0.5 ist
# wie das bei mir der fall ist (bei schroedinger ml), dann ist es am
# besten, diese basis schwingung manuell einzutragenm und die oberschwingungen
# durch die fft zu ermitteln.
# PROBLEME: y-intercept und wie vermeide ich, dass eines der As und bs zur
# Grundschwingung gehört
                         
plt.figure(1)
plt.subplot(3,1,1)
plt.grid(1)
plt.xlabel('x')
plt.ylabel('signal')
plt.plot(x, sgnl, label='signal')
plt.legend(loc='upper right')

plt.subplot(3,1,2)
plt.grid(1)
plt.xlabel('frequency [Hz]')
plt.ylabel('absolute spectrum\n(processed)')
plt.plot(freqs[realFreqs], fft_phys[realFreqs], '.', label='processed FFT')
plt.legend(loc='upper right')

plt.subplot(3,1,3)
plt.grid(1)
plt.xlabel('frequency [Hz]')
plt.ylabel('spectrum\n(unprocessed)')
plt.plot(freqs, fft_vals, '.', label='unprocessed FFT')
plt.legend(loc='upper right')

plt.subplot(3,1,1)
plt.grid(1)
plt.plot(x, sgnl_synth, 'x', label='signal (synthesized)')
plt.legend(loc='upper right')


#ERKENNTNIS: WERT BEI 0 = n * d (BEI PLOT VON FFT ÜBER FREQ, beides unprocessed und ohne phase shift)

# plt.figure(2)
# plt.plot(fft_vals.real, 'o')

# %%
plt.plot(np.linspace(0, 127, 128)[0::2][0:63], (sgnl_synth[0:63]-0.74)*2)
plt.plot(trainy[360])

#%%

dom = np.linspace(0, 4*np.pi, 100)
liste = []
for k in dom:
    ana = fft(10 * np.sin(2 * w * x + k) + 0)
    liste.append(ana[2])

plt.grid(1)
plt.plot(dom, liste, 'o')
#%%
# synthesized signal assembly
harm1 = 1*np.sin(0.65 * w*x)
harm2 = 2*np.sin(0.345 * w*x)
harm3 = 3*np.sin(0.225 * w*x)
harm4 = 4*np.sin(0.05 * w*x)
harm5 = 5*np.sin(0.03 * w*x)

sgnl_synth = harm1 + harm2+ harm3 + harm4 + harm5

plt.subplot(3,1,3)
plt.grid(1)
plt.xlabel('x')
plt.ylabel('signal (synthesized)')
plt.plot(x, sgnl_synth, label='signal')
plt.legend(loc='upper right')