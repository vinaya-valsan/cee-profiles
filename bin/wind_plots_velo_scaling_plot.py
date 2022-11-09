import yt
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import module
import numpy as np
import  scipy
from scipy.optimize import least_squares
from dataclasses import dataclass
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
print(matplotlib.rcParams)
G=6.67e-8
Rsun = 7.0e10
dPeriod = 0.5e16
lim = dPeriod / 2. * 1.0001
hbox = np.array([[-lim,lim],[-lim,lim],[-lim,lim]])
au = 1.496e+13 # cm to au
km_per_s = 1e5 # cm/s to km/s
kg_km_per_s = 1e3*1e5 # g*cm/s to kg*km/s


t1=300
#t1=200
R1 = 2.5e14
ts_seed = 110
ts_min = 100
#ts_min = 50
ts_max = t1-1

FileID=[]
nfiles = 12
interval= 50
for i in range(0, nfiles) :
	num = t1 + i * interval
	FileID.append("{0:06d}".format(num))


print(FileID)
star_data={}
for x in FileID:
	try:
		star_data[x] = module.CreateDataStructure(x)
	except:
		continue

def ComputeRadialV_fit(ts,fileid, data_dict):
    Vr = data_dict[fileid]['Gas_RadialV']
    R = data_dict[fileid]['Gas_radius']
    Gas_mass = data_dict[fileid]['Gas_mass']
    current_t = data_dict[fileid]['current_time']
    Rs = R*((t1-ts)/(current_t.v-ts))
    
    radius_bins = 10.**np.arange(11,15.5,0.05)
    X,Y = module.CreateRadialProfile(Rs, Vr,bins_x = radius_bins,weight_field=Gas_mass)
    #x_sel = X[X<3e14]
    #y_sel = Y[(X<3e14)]
    #max_v_ind = np.argmax(y_sel)
    #r_for_max_v = x_sel[max_v_ind]
    #mask = (X<r_for_max_v)*(X>2e13)
    mask = (X<3e14)*(X>5e13)
    #mask*= (Y>=0.)
    x_masked = X[mask]
    y_masked = Y[mask]
    alpha, y0  = np.polyfit(x_masked, y_masked, deg=1)
    beta, lgy0  = np.polyfit(np.log10(x_masked), np.log10(y_masked), deg=1)
    print('fileid=%s,ts=%s, alpha=%s, v0=%s, beta=%s, lgv0=%s'%(fileid,ts[0],alpha,y0,beta,lgy0))
    return alpha,y0, beta, lgy0


def get_sigma(ts, data_dict=star_data):
    fileids = data_dict.keys()
    V0_array=[]
    alpha_array=[]
    beta_array =[]
    LogV0_array =[]
    for fileid in fileids:
        alpha,v0, beta, lgv0 = ComputeRadialV_fit(ts,fileid, data_dict=star_data)
        V0_array.append(v0)
        alpha_array.append(alpha)
        beta_array.append(beta)
        LogV0_array.append(lgv0)
    sigma_v = np.std(V0_array)/np.mean(V0_array)
    sigma_alpha = np.std(alpha_array)/np.mean(alpha_array)
    sigma = np.sqrt(sigma_v**2.+sigma_alpha**2)
    sigma_lgv = np.std(LogV0_array)/np.mean(LogV0_array)
    sigma_beta = np.std(beta_array)/np.mean(beta_array)
    sigma_lg = np.sqrt(sigma_lgv**2.+sigma_beta**2) 
    print('sigma =%s, sigma_lg=%s'%(sigma,sigma_lg))
    return sigma_lg




res = least_squares(get_sigma, ts_seed,bounds=(ts_min, ts_max),diff_step = 0.05)
ts_optimum=(res.x)[0]



print('optimal ts', ts_optimum)
FileID = FileID[0:10:2]
print(FileID)
#FileID = ['000300','000400','000500','000600','000700','000800','000900']#,'001000']#,'001500','002000']
#labellist = ['300d','400d','500d','600d','700d','800d','900d']#,'1000d']#,'1500d','2000d']
FileID = ['000300','000800','002000']
labellist = [r'$300\, \rm d$',r'$800\, \rm d$',r'$2000\, \rm d$']
rho_atm = 1e-14 


#--------------------
ds = yt.load('star.out.000001', bounding_box=hbox)
ad = ds.all_data()

Rgiant = 52*Rsun
coord = ad[("Gas", "Coordinates")]
r = np.linalg.norm(coord, axis=-1)
Nstar = r[r<Rgiant].size

print('Nstar',Nstar)
#--------------------------


fig1,ax1 = plt.subplots()#figsize=(13.5,8.8))
fig2,ax2 = plt.subplots()#figsize=(13.5,8.8))
fig3,ax3 = plt.subplots()#figsize=(13.5,8.8))
for inde,fileid in enumerate(FileID):
	try:
		t = star_data[fileid]['current_time']
	except:
		star_data[fileid] = module.CreateDataStructure(fileid)	
		t = star_data[fileid]['current_time']
	Gas_radius = star_data[fileid]['Gas_radius'][:Nstar]	
	Gas_density = star_data[fileid]['Gas_density'][:Nstar]
	R_env = R1*(t.v-ts_optimum)/(t1-ts_optimum)
	eta = Gas_radius/R_env
	Scaled_density= Gas_density*((t.v-ts_optimum)/(t1-ts_optimum))**3
	ind = np.argwhere(eta>1)
	Scaled_density[ind] = rho_atm
	Scaled_radius = Gas_radius*((t1-ts_optimum)/(t.v-ts_optimum))
	Gas_volume = star_data[fileid]['Gas_volume'][:Nstar]
	Gas_mass = star_data[fileid]['Gas_mass'][:Nstar]
	Gas_RadialV = abs(star_data[fileid]['Gas_RadialV'][:Nstar])
	Gas_TangentialV = abs(star_data[fileid]['Gas_TangentialV'][:Nstar])
	Gas_temperature = star_data[fileid]['Gas_temperature'][:Nstar]




	radius_bins = 10.**np.arange(11,15.5,0.05)

	x,y = module.CreateRadialProfile(Scaled_radius, Scaled_density,bins_x = radius_bins,weight_field=Gas_volume)
	ax1.plot(x/Rsun,y,label=labellist[inde])
        
	#x,y = module.CreateRadialProfile(Gas_radius, Gas_density,bins_x = radius_bins,weight_field=Gas_volume)
	#ax[1,0].plot(x,y, label=fileid)

	
	x,y = module.CreateRadialProfile(Scaled_radius, Gas_RadialV,bins_x = radius_bins,weight_field=Gas_mass)
	ax2.plot(x/Rsun,y/1e5,label=labellist[inde])
	
	#x,y = module.CreateRadialProfile(Gas_radius, Gas_RadialV,bins_x = radius_bins,weight_field=Gas_mass)
	#ax[1,1].plot(x,y, label=fileid)


	#	
	#x,y = module.CreateRadialProfile(Scaled_radius, Gas_TangentialV,bins_x = radius_bins,weight_field=Gas_mass)
	#ax[0,2].plot(x,y, label=fileid)

	#x,y = module.CreateRadialProfile(Gas_radius, Gas_TangentialV,bins_x = radius_bins,weight_field=Gas_mass)
	#ax[1,2].plot(x,y, label=fileid)


	x,y = module.CreateRadialProfile(Scaled_radius, Gas_temperature,bins_x = radius_bins,weight_field=Gas_mass)
	ax3.plot(x/Rsun,y, label=fileid)

	#x,y = module.CreateRadialProfile(Gas_radius, Gas_temperature,bins_x = radius_bins,weight_field=Gas_mass)
	#ax[1,3].plot(x,y, label=fileid)
	#	
	#x,y = module.CreateRadialProfile(Scaled_radius, Gas_RadialV/Gas_TangentialV ,bins_x = radius_bins,weight_field=Gas_mass)
	#ax[0,4].plot(x,y, label=fileid)

	#x,y = module.CreateRadialProfile(Gas_radius, Gas_RadialV/Gas_TangentialV ,bins_x = radius_bins,weight_field=Gas_mass)
	#ax[1,4].plot(x,y, label=fileid)




#fig,ax = plt.subplots(2,5,figsize=(40,10))
num = t1#10 * round(ts_optimum/10)
fileid =  "{0:06d}".format(num)
r0=Rsun#1e12
star_data[fileid] = module.CreateDataStructure(fileid)
t = star_data[fileid]['current_time']
Gas_radius = star_data[fileid]['Gas_radius'][:Nstar]
Gas_density = star_data[fileid]['Gas_density'][:Nstar]
R_env = R1*(t.v-ts_optimum)/(t1-ts_optimum)
eta = Gas_radius/R_env
Scaled_density= Gas_density*((t.v-ts_optimum)/(t1-ts_optimum))**3
ind = np.argwhere(eta>1)
Scaled_density[ind] = rho_atm
Scaled_radius = Gas_radius*((t1-ts_optimum)/(t.v-ts_optimum))
Gas_volume = star_data[fileid]['Gas_volume'][:Nstar]
Gas_mass = star_data[fileid]['Gas_mass'][:Nstar]
Gas_RadialV = np.abs(star_data[fileid]['Gas_RadialV'][:Nstar])
#Gas_RadialV = star_data[fileid]['Gas_RadialV'][:Nstar]
Gas_TangentialV = np.abs(star_data[fileid]['Gas_TangentialV'][:Nstar])
Gas_temperature = star_data[fileid]['Gas_temperature'][:Nstar]
Scaled_Temperature = Gas_temperature*((t.v-ts_optimum)/(t1-ts_optimum))**2
t_atm = 1e4
Scaled_Temperature[ind] = t_atm


radius_bins = 10.**np.arange(11,15.8,0.05)

x,y = module.CreateRadialProfile(Scaled_radius, Scaled_density,bins_x = radius_bins,weight_field=Gas_volume)
mask = (x<2e14)*(x>7e13)
x_masked = x[mask]
y_masked = y[mask]
beta, lgy0  = np.polyfit(np.log10(x_masked), np.log10(y_masked), deg=1)
x_plot = np.arange(13,14.5,0.05)
y_plot = 10**(lgy0+beta*x_plot)
#ax[0,0].set_title('lgy0=%0.2f, beta=%0.2f'%(lgy0,beta))
ax1.plot(10**(x_plot)/Rsun,y_plot,'k',label = r'$\rho_s \propto r_s^{-3.2}$')#,markersize=50)
#ax[0,0].plot(x,y)
print(f'density fit: rho = rho0*(r/r0)**beta: rho0 = {10**lgy0*r0**beta}, beta = {beta}')   


#x,y = module.CreateRadialProfile(Gas_radius, Gas_density,bins_x = radius_bins,weight_field=Gas_volume)
#ax[1,0].plot(x,y)


x,y = module.CreateRadialProfile(Scaled_radius, Gas_RadialV,bins_x = radius_bins,weight_field=Gas_mass)
mask = (x<2e14)*(x>7e13)
x_masked = x[mask]
y_masked = y[mask]
beta, lgy0  = np.polyfit(np.log10(x_masked), np.log10(y_masked), deg=1)
x_plot = np.arange(13,14.5,0.05)
y_plot = 10**(lgy0+beta*x_plot)
#ax[0,1].set_title('lgy0=%0.2f, beta=%0.2f'%(lgy0,beta))
ax2.plot(10**(x_plot)/Rsun,y_plot/1e5,'k',label = r'$v_{r} \propto r_s^{0.96}$')#,markersize=50)
print(f'velocity fit: vr = vr0*(r/r0)**beta: vr0 = {10**lgy0*r0**beta/1e5}, beta = {beta}')

#ax[0,1].plot(x,y)
#
#x,y = module.CreateRadialProfile(Gas_radius, Gas_RadialV,bins_x = radius_bins,weight_field=Gas_mass)
#ax[1,1].plot(x,y)
#
#x,y = module.CreateRadialProfile(Scaled_radius, Gas_TangentialV,bins_x = radius_bins,weight_field=Gas_mass)
#ax[0,2].plot(x,y)
#
#x,y = module.CreateRadialProfile(Gas_radius, Gas_TangentialV,bins_x = radius_bins,weight_field=Gas_mass)
#ax[1,2].plot(x,y)
#
x,y = module.CreateRadialProfile(Scaled_radius, Scaled_Temperature,bins_x = radius_bins,weight_field=Gas_mass)
mask = (x>40.*Rsun)*(x<80.*Rsun)
x_masked = x[mask]
y_masked = y[mask]
beta, lgy0  = np.polyfit(np.log10(x_masked), np.log10(y_masked), deg=1)
x_plot = np.arange(12.4,13.,0.05)
y_plot = 10**(lgy0+beta*x_plot)
ax3.plot(10**(x_plot)/Rsun,y_plot,'k',label = r'$T \propto r_s^{-2.71}$')#,markersize=50)
print(f'temp fit: T = T0*(r/r0)**beta: T0 = {10**lgy0*r0**beta}, beta = {beta}')
#ax[0,3].set_title('lgy0=%0.2f, beta=%0.2f'%(lgy0,beta))

#ax[0,3].plot(10**(x_plot),y_plot)
#ax[0,3].plot(x,y)
#print(f'temperature fit: t = t0*(r/r0)**beta: y0 = {10**lgy0*r0**beta}, beta = {beta}')    
#
#x,y = module.CreateRadialProfile(Gas_radius, Gas_temperature,bins_x = radius_bins,weight_field=Gas_mass)
#ax[1,3].plot(x,y)
#
#
#x,y = module.CreateRadialProfile(Scaled_radius, np.abs(Gas_RadialV)/Gas_TangentialV,bins_x = radius_bins,weight_field=Gas_mass)
#mask = (x<2e14)*(x>7e13)
#x_masked = x[mask]
#y_masked = y[mask]
#beta, lgy0  = np.polyfit(np.log10(x_masked), np.log10(y_masked), deg=1)
#x_plot = np.arange(11,15.8,0.05)
#y_plot = 10**(lgy0+beta*x_plot)
#ax[0,4].set_title('lgy0=%0.2f, beta=%0.2f'%(lgy0,beta))
#ax[0,4].plot(10**(x_plot),y_plot)
#ax[0,4].plot(x,y)
#
#
#x,y = module.CreateRadialProfile(Gas_radius, np.abs(Gas_RadialV)/Gas_TangentialV,bins_x = radius_bins,weight_field=Gas_mass)
#ax[1,4].plot(x,y)

#
#ax[0,0].legend()
#ax[0,0].set_xlabel('Scaled_Radius')
#ax[0,0].set_ylabel('Scaled_Density')
#ax[0,0].set_xscale('log')
#ax[0,0].set_yscale('log')
#
#
#ax[1,0].legend()
#ax[1,0].set_xlabel('Radius')
#ax[1,0].set_ylabel('Density')
#ax[1,0].set_xscale('log')
#ax[1,0].set_yscale('log')
#
#ax[0,1].set_xlabel('Scaled_Radius')
#ax[0,1].set_ylabel('Radial_Vel')
#ax[0,1].set_xscale('log')
#ax[0,1].set_yscale('log')
#
#ax[1,1].set_xlabel('Radius')
#ax[1,1].set_ylabel('Radial_Vel')
#ax[1,1].set_xscale('log')
#ax[1,1].set_yscale('log')
#
#
#ax[0,2].set_xlabel('Scaled_Radius')
#ax[0,2].set_ylabel('Tangential_Vel')
#ax[0,2].set_xscale('log')
#ax[0,2].set_yscale('log')
#
#ax[1,2].set_xlabel('Radius')
#ax[1,2].set_ylabel('Tangential_Vel')
#ax[1,2].set_xscale('log')
#ax[1,2].set_yscale('log')
#
#ax[0,3].set_xlabel('Scaled_Radius')
#ax[0,3].set_ylabel('Scaled_Temperature')
#ax[0,3].set_xscale('log')
#ax[0,3].set_yscale('log')
#
#ax[1,3].set_xlabel('Radius')
#ax[1,3].set_ylabel('Temperature')
#ax[1,3].set_xscale('log')
#ax[1,3].set_yscale('log')
#
#ax[0,4].set_xlabel('Scaled_Radius')
#ax[0,4].set_ylabel('Radial_Vel/Tangential_Vel')
#ax[0,4].set_xscale('log')
#ax[0,4].set_yscale('log')
#
#ax[1,4].set_xlabel('Radius')
#ax[1,4].set_ylabel('Radial_Vel/Tangential_Vel')
#ax[1,4].set_xscale('log')
#ax[1,4].set_yscale('log')
#
#plt.savefig('abs_singlefit_fileid-'+fileid+'starcut.png')

ax1.legend(loc='upper right')
ax1.set_xlabel(r'$r_s~(R_{\odot})$')#,fontsize=28)
ax1.set_ylabel(r'$\rho_s~(\rm g \, cm^{-3})$')#,fontsize=28)
ax1.tick_params(axis="y")#, labelsize=28)
ax1.tick_params(axis="x")#, labelsize=28)
ax1.set_xscale('log')
ax1.set_yscale('log')
fig1.tight_layout()
fig1.savefig('ScaledDensity.pdf')#,dpi=50)

#ax[1,0].legend()
#ax[1,0].set_xlabel('Radius')
#ax[1,0].set_ylabel('Density')
#ax[1,0].set_xscale('log')
#ax[1,0].set_yscale('log')

ax2.legend(loc='upper left')
ax2.set_xlabel(r'$r_s~(R_{\odot})$')#,fontsize=28)
ax2.set_ylabel(r'$v_{r}~(\rm km \,s^{-1})$')#,fontsize=28)
ax2.tick_params(axis="y")#, labelsize=28)
ax2.tick_params(axis="x")#, labelsize=28)
ax2.set_xscale('log')
ax2.set_yscale('log')
fig2.tight_layout()
fig2.savefig('RadialV.pdf')#,dpi=50)

ax3.legend(loc='upper left')
ax3.set_xlabel(r'$r_s~(R_{\odot})$')#,fontsize=28)
ax3.set_ylabel(r'$T~(\rm K)$')#,fontsize=28)
ax3.tick_params(axis="y")#, labelsize=28)
ax3.tick_params(axis="x")#, labelsize=28)
ax3.set_xscale('log')
ax3.set_yscale('log')
fig3.tight_layout()
fig3.savefig('Temperature.pdf')#,dpi=50)



