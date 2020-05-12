import numpy as np
import matplotlib.pyplot as plt

#%%
def f(t):
    return 2*np.cos(t) + np.sin(t) + np.sin(4*t)

t2 = np.arange(0.0, 5.0, 0.02)

plt.figure()
plt.grid(1)
plt.ylabel('$\Psi$(x)', fontsize=32)
plt.xlabel('x', fontsize=32)

plt.plot(t2, f(t2), linewidth=3, c='b', label='$\Psi$(x)')

plt.legend(fontsize=22)

#%%
def f(t):
    return 2*np.cos(t)

t2 = np.arange(0.0, 5.0, 0.02)

plt.figure()
plt.grid(1)

plt.plot(t2, f(t2), linewidth=3, c='m')

#%%
def f(t):
    return np.sin(t)

t2 = np.arange(0.0, 5.0, 0.02)
plt.grid(1)
plt.plot(t2, f(t2), linewidth=3, c='c')

#%%
def f(t):
    return np.sin(4*t)

t2 = np.arange(0.0, 5.0, 0.02)
plt.grid(1)
plt.plot(t2, f(t2), linewidth=3, c='y')

#%%
plt.savefig('/home/domi/Dokumente/BScPresentation/psicomposed.png', orientation='landscape', transparent=True)
