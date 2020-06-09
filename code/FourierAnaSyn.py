import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

#%%
path = '/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/'

#%%
bins = 128
seedmax = 20
trainy = []
validy = []

for i in range(seedmax):
    with open(path+'test_out/test_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainy.append([float(num) for num in row])
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
# increase spatial resolution of wavefunction
# by interpolation for more precise FFT
index = np.random.randint(0, 9600)
x = np.linspace(0, 126, 127)
y = trainy[index]
xvals = np.linspace(0, 126, 1000)
yinterp = np.interp(xvals, x, y)

#%%
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
for f in range(1, 12):
    Asynth = fft_phys[f]
    re = fft_vals.real[f]
    im = fft_vals.imag[f]
    s =  cosShift(re, im)*np.pi/180
    sgnl_synth += Asynth * np.cos(f * w * x + s)

# plot analyzed and synthesized wavefunctions
plt.grid(1)
plt.plot(sgnl[0:1000], linewidth=2, label='original Signal')
plt.plot(sgnl_synth[0:1000], linewidth=2, label='synthesized Signal')
plt.legend()
plt.tight_layout()

# plot all information
plt.figure(2)
plt.subplot(511)
plt.grid(1)
plt.title('Signal extended as odd function')
plt.plot(sgnl)

plt.subplot(512)
plt.grid(1)
plt.title('Synthesized Signal')
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