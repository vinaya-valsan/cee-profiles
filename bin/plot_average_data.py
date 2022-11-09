import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib import rc
import numpy as np
import matplotlib
#plt.switch_backend('agg')


params = {'backend': 'agg',
          'axes.labelsize': 24, # fontsize for x and y labels (was 10)
          'axes.titlesize': 24,
          'axes.labelweight': 'heavy',
          'legend.fontsize': 18, # was 10
          'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'text.usetex': True,
          'figure.figsize': [8,6],
          'figure.dpi': 300,
          'savefig.dpi': 300,
          'font.family': 'serif',
          'font.serif': ['Times'],
          'font.weight':'heavy',
          'font.size': 24,
          'lines.linewidth': 2
}


matplotlib.rcParams.update(params)

#print((matplotlib.rcParams.keys))

Time,AverageDensity,AverageDensity_VolWgt,AverageDensity_unb,AverageDensity_VolWgt_unb= np.loadtxt('average_data.txt',skiprows=1, unpack=True)

t0=100
th = Time-t0
mask = (th>(2100-t0))
a,b=np.polyfit(th[mask]**(-3), AverageDensity_VolWgt[mask],deg=1)
t_sample = np.linspace(0,max(Time),500)

#plt.figure()#figsize=(11,8))
#plt.plot(t_sample, b+a*t_sample**(-3),'k--')
#plt.plot(Time, AverageDensity_VolWgt)#,'g')
#plt.xlabel(r'$t~(\rm d)$')#,fontsize=18)
#plt.ylabel(r'$\bar{\rho}~(\rm g\,cm^{-3} )$')#,fontsize=18)
#plt.yscale('log')
#plt.tight_layout()
#plt.savefig('AverageDensity_fit.pdf')#,dpi=500)


fig, ax = plt.subplots()#figsize=(11,8))
ax.plot(t_sample, b+a*t_sample**(-3),'k--')
ax.plot(Time, AverageDensity_VolWgt)#,'g')
ax.set_xlabel(r'$t~(\rm d)$')#,fontsize=18)
ax.set_ylabel(r'$\bar{\rho}~(\rm g\,cm^{-3} )$')#,fontsize=18)
ax.set_yscale('log')
fig.tight_layout()
fig.savefig('AverageDensity_fit.png')#,dpi=500)
fig.savefig('AverageDensity_fit.pdf')#,dpi=500)

'''
t0=300
th = Time-t0
mask = (th>(2000-t0))
a,b=np.polyfit(th[mask]**(-3), AverageDensity_VolWgt_unb[mask],deg=1)
t_sample = np.linspace(0,max(Time),500)
plt.figure(figsize=(10,8))
plt.plot(t_sample, b+a*t_sample**(-3),'k--')
plt.plot(Time, AverageDensity_VolWgt_unb,'g')
#plt.grid()
plt.yscale('log')
plt.xlabel('t(d)',fontsize=16)
plt.ylabel(r'$\bar{\rho}$($gcm^{-3}$)',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('AverageDensity_unb_fit.png',dpi=500)
'''
