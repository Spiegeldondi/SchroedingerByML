import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

def cosShift(re, im):
    shift = np.arctan2(im, re) * 180/np.pi
    shift = np.floor(shift)
    if shift < 0:
        shift += 360
    return shift

def fourierDecomp(wavefunction):
    length = len(wavefunction)
    # increase spatial resolution of wavefunction
    # by interpolation for more precise FFT
    index = np.random.randint(0, 9600)
    x = np.linspace(0, length-1, length)
    y = wavefunction
    xvals = np.linspace(0, length-1, 1000)
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
    
    A_list = []
    f_list = []
    s_list = []
    
    Afs = []
    
    for f in range(1, 9):
        Asynth = fft_phys[f]
        re = fft_vals.real[f]
        im = fft_vals.imag[f]
        s =  cosShift(re, im)*np.pi/180
        sgnl_synth += Asynth * np.cos(f * w * x + s)
        
        A_list.append(Asynth)
        f_list.append(f)
        s_list.append(s)
        
        Afs.append(f)
        Afs.append(Asynth)
        Afs.append(s)

    return Afs

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
FourierInfo = fourierDecomp(trainy[79])

T = 127
v = 1/T
# angular frequency
w = 2*np.pi*v
# spatial resolution and domain'
n = 254
x = np.linspace(0, T, n)

sgnl_synth = 0    
   
sgnl_synth += FourierInfo[1] * np.cos(FourierInfo[0] * w * x + FourierInfo[2])
sgnl_synth += FourierInfo[4] * np.cos(FourierInfo[3] * w * x + FourierInfo[5])
sgnl_synth += FourierInfo[7] * np.cos(FourierInfo[6] * w * x + FourierInfo[8])
sgnl_synth += FourierInfo[10] * np.cos(FourierInfo[9] * w * x + FourierInfo[11])
sgnl_synth += FourierInfo[13] * np.cos(FourierInfo[12] * w * x + FourierInfo[14])
sgnl_synth += FourierInfo[16] * np.cos(FourierInfo[15] * w * x + FourierInfo[17])
sgnl_synth += FourierInfo[19] * np.cos(FourierInfo[18] * w * x + FourierInfo[20])
sgnl_synth += FourierInfo[22] * np.cos(FourierInfo[21] * w * x + FourierInfo[23])

plt.plot(trainy[79])
plt.plot(sgnl_synth[:127])
    

#%%
# plot analyzed and synthized wavefunctions
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