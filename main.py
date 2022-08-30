import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import scipy.stats as sst
from ns_cavity import ComplexNSCavity

##Driven Cavity Studies
hs = np.array([0.01, 0.01, 0.01], dtype=np.longdouble)
uws = np.array([1.0, 0, 0, 0, 0, 0], dtype=np.longdouble)
Re = 100
tstop = 1.0

sim1 = ComplexNSCavity(hs, tstop, uws, Re, sor=True, dtype=np.longdouble)
sim2 = ComplexNSCavity(hs, tstop, uws, Re, sor=False, dtype=np.longdouble)

plt.close('all')
t1 = np.arange(len(sim1.vniter))
t2 = np.arange(len(sim2.vniter))

f = plt.figure(figsize=(10, 10))
plt.plot(t1, sim1.vniter, c='C0', linestyle='-', label='Vorticity Iterations, SOR')
plt.plot(t1, sim1.sniter, c='C0', linestyle='--', label='Stream Iterations, SOR')
plt.plot(t2, sim2.vniter, c='C1', linestyle='-', label='Vorticity Iterations, GS')
plt.plot(t2, sim2.sniter, c='C1', linestyle='--', label='Stream Iterations, GS')
plt.legend()
plt.title('Iterations to Convergence For Cavity Sim')
plt.xlabel(r'$N_{step}$')
plt.ylabel(r'$N_{iter}$')
plt.savefig('vs_iter.png')

plt.close('all')
t1 = np.arange(len(sim1.vniter))
t2 = np.arange(len(sim2.vniter))

f = plt.figure(figsize=(10, 10))
plt.plot(t1, sim1.linf_dt, c='C0', linestyle='-', label=r'$|\omega(n)-\omega_{n-1}|_{\infty}$, SOR')
plt.plot(t2, sim2.linf_dt, c='C1', linestyle='-', label=r'$|\omega_{n}-\omega_{n-1}|_{\infty}$, GS')
plt.xlabel(r'$N_{step}$')
plt.legend()
plt.title('Iterations to Convergence For Cavity Sim')
plt.savefig('iter.png')
