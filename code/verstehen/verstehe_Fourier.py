import csv
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

#%%
bins = 128
seedmax = 20 # opens seed files 0 - 19. Lost too much data due to kernel crashes, so these got broken up
trainx = []
trainy = []
validx = []
validy = []

#%%
path = '/home/domi/Dokumente/SchroedingerByML/potentials/A_original_potentials/'

# %% This is not a ... pythonic [barf]... way of reading data, but python is stupid about pointers, so deal with it
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
# period and frequency
T = 1
v = 1/T

# angular frequency
w = 2*np.pi*v

# spatial resolution and domain
n = 127
x = np.linspace(0, T, n)

# parameters of periodic function
A = 10
f = 4
phi = np.pi*0
d = 20

# signal components
# sig1 = A * np.sin(f * w * x + phi) + d
# sig2 = 15 * np.sin(2 * w * x + phi) + 30
# sig3 = 25 * np.sin(6 * w * x + phi) + 7
sig1 = np.array(trainy[360])

# signal assambly
sgnl = sig1# + sig2 + sig3

# frequency domain
freqs = fftfreq(n) * n

# true physical frequency domain
realFreqs = freqs > 0

# fft
fft_vals = fft(sgnl)

# true physical fft
fft_phys = 2.0*np.abs(fft_vals/n)

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

sgnl_synth = yintercept+( A1 * np.sin(b1 * w * x + phi1) + A2 * np.sin(b2 * w * x + phi2) + A3 * np.sin(b3 * w * x + phi3) + A4 * np.sin(b4 * w * x + phi3) )
# sgnl_synth = yintercept + ( A1 * np.sin(b1 * w * x + phi1))
                         
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