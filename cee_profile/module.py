import yt
import numpy as np

G=6.67e-8

def CreateRadialProfile(data_x, data_y, bins_x, weight_field=None):
    hist_y = np.zeros(len(bins_x)-1)
    hist_x = [0.5*(bins_x[i]+bins_x[i+1]) for i in range(len(bins_x)-1)]
    for i in range(len(bins_x)-1):
        bin_up = bins_x[i+1]
        bin_low = bins_x[i]
        indices_inside = np.where((data_x>=bin_low) & (data_x<bin_up))

        if weight_field is None:
             hist_y[i] = sum(data_y[indices_inside])
        else:
            hist_y[i] = sum(data_y[indices_inside]*weight_field[indices_inside])
            if sum(weight_field[indices_inside])==0.:
                hist_y[i] = hist_y[i]/1e-10
            else:
                hist_y[i] = hist_y[i]/sum(weight_field[indices_inside])
    mid_r = np.array(hist_x)
    return mid_r, np.array(hist_y)


def lagrangian_mass(radius,gas_mass, radial_bins):
    mid_r = []
    mass_per_bin = []
    lag_mass = []
    for bin_id in range(1,len(radial_bins)):
        min_r = radial_bins[bin_id-1]
        max_r = radial_bins[bin_id]
        mid_r.append(0.5*(min_r+max_r))
        mask = (radius<max_r)*(radius>min_r)
        mass_per_bin.append(sum(gas_mass[mask]))
        lag_mass.append(sum(mass_per_bin))
    return np.array(mid_r), np.array(lag_mass)



def CreateDataStructure(fileid):
    data_dict={}
    filename = "star.out."+fileid

    ds = yt.load(filename)
    ad = ds.all_data()



    DM_vel = ad[("DarkMatter", "Velocities")].v
    DM_mass = ad[("DarkMatter", "Mass")].v
    DM_coord = ad[("DarkMatter", "Coordinates")].v
    Gas_coord_i = ad[('Gas', 'Coordinates')].v
    Gas_density = ad[('Gas', 'Density')].v
    Gas_mass = ad[('Gas', 'Mass')].v
    Gas_velocity = ad[('Gas', 'Velocities')].v
    Gas_temperature = ad[('Gas', 'Temperature')].v	
    Gas_volume = Gas_mass/Gas_density

    t=ds.current_time.in_units('day')

    ## Radial and Tangential Velocity
    cm_coord = np.average(DM_coord, axis=0, weights=DM_mass)
    cm_vel = np.average( DM_vel, axis=0, weights=DM_mass)
    Gas_coord = Gas_coord_i - cm_coord
    Gas_radius = np.linalg.norm(Gas_coord,axis=-1)
    r_hat = np.divide(Gas_coord,Gas_radius[:,np.newaxis])
    vel_rel =Gas_velocity - cm_vel[np.newaxis,:]
    RadialV_vec = np.multiply(np.sum(r_hat*vel_rel,axis=-1)[:,np.newaxis],r_hat)
    Gas_RadialV = np.linalg.norm(RadialV_vec,axis=-1)
    TangentialV_vec = vel_rel- RadialV_vec
    Gas_TangentialV = np.linalg.norm(TangentialV_vec,axis=-1)

    # Energy
    Gas_phi = Gas_mass*ad[('Gas', 'Phi')].v   
    Gas_ie = Gas_mass*ad[('Gas', 'ie')].v/G 
    Gas_ke = 0.5*Gas_mass*Gas_RadialV**2. 
    Gas_TotalE = Gas_phi+Gas_ie+Gas_ke
     

    # Save in cgs, current_time in days
    data_dict['Gas_RadialV'] = Gas_RadialV*np.sqrt(G)
    data_dict['Gas_velocity'] = Gas_velocity*np.sqrt(G)
    data_dict['Gas_radius'] = Gas_radius
    data_dict['current_time'] = t#/np.sqrt(G)
    data_dict['Gas_density'] = Gas_density
    data_dict['Gas_mass'] = Gas_mass
    data_dict['Gas_volume'] = Gas_volume
    data_dict['Gas_TangentialV'] = Gas_TangentialV*np.sqrt(G)
    data_dict['Gas_phi'] = Gas_phi*G
    data_dict['Gas_ie'] = Gas_ie*G
    data_dict['Gas_ke'] = Gas_ke*G
    data_dict['Gas_TotalE'] = Gas_TotalE*G
    data_dict['Gas_temperature'] = Gas_temperature
    return data_dict


