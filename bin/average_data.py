import yt
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import  scipy
from scipy.optimize import least_squares
from dataclasses import dataclass
import module
####################### SETTINGS ############################

nfiles = 1000            # number of output files to analyze
first =  1              # start at this file
interval = 5           # interval between output files
nref = 32              # resolution of pictures
                       # (1 is highest, 64 is lowest)

do_marks = True        # mark positions of core and companion
colormap = 'inferno'   # which colormap is used
colormap_lower = 6.0e4 # lower limit on density for colormap
colormap_upper = 4.0e8 # upper limit on density for colormap
dataname = '../star.out.' # prefix on output files

#############################################################

filename = []
for i in range(0, nfiles) :
    num = first + i * interval
    filename.append("{0:06d}".format(num))

G=6.67e-8
Rsun = 7.0e10
dPeriod = 0.5e16
lim = dPeriod / 2. * 1.0001
hbox = np.array([[-lim,lim],[-lim,lim],[-lim,lim]])
au = 1.496e+13 # cm to au
km_per_s = 1e5 # cm/s to km/s 
kg_km_per_s = 1e3*1e5 # g*cm/s to kg*km/s 
#--------------------
ds = yt.load('../star.out.000001', bounding_box=hbox)
ad = ds.all_data()

Rgiant = 52*Rsun
coord = ad[("Gas", "Coordinates")]
r = np.linalg.norm(coord, axis=-1)
Nstar = r[r<Rgiant].size
print(Nstar)
#---------------

with open('average_data.txt','w') as ff:
	ff.write('# Time(day) AverageDensity(cgs) AverageDensity_VolWgt(cgs) AverageDensity_unb(cgs) AverageDensity_VolWgt_unb(cgs)')
	ff.write('\n')
for x in filename:
    try:
        data = module.CreateDateStructure(x)
    except:
        continue
    Gas_radius = data['Gas_radius'][:Nstar]
    Gas_RadialV = data['Gas_RadialV'][:Nstar]
    Gas_TangentialV = data['Gas_TangentialV'][:Nstar]
    Gas_density = data['Gas_density'][:Nstar]
    Gas_TotalE = data['Gas_TotalE'][:Nstar]
    Gas_mass = data['Gas_mass'][:Nstar]/1e3	
    Gas_volume = data['Gas_volume'][:Nstar]

    AverageDensity = np.average(Gas_density)
    AverageDensity_VolWgt = np.average(Gas_density, weights=Gas_volume)
    Time = data['current_time'].v


    inds = np.where(Gas_TotalE>0.)[0]

    AverageDensity_unb = np.average(Gas_density[inds])
    AverageDensity_VolWgt_unb = np.average(Gas_density[inds], weights=Gas_volume[inds])
    with open('average_data.txt','a') as ff:
        line = f'{Time} {AverageDensity} {AverageDensity_VolWgt} {AverageDensity_unb} {AverageDensity_VolWgt_unb}'
        ff.write(line)
        ff.write('\n')

