import os
import yt
import logging
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


class Constants:
    G = 6.67e-8
    Rsun = 7.0e10


def CreateRadialProfile(data_x, data_y, bins_x, weight_field=None):
    hist_y = np.zeros(len(bins_x) - 1)
    hist_x = [
        0.5 * (bins_x[i] + bins_x[i + 1]) for i in range(len(bins_x) - 1)
    ]
    for i in range(len(bins_x) - 1):
        bin_up = bins_x[i + 1]
        bin_low = bins_x[i]
        indices_inside = np.where((data_x >= bin_low) & (data_x < bin_up))

        if weight_field is None:
            hist_y[i] = sum(data_y[indices_inside])
        else:
            hist_y[i] = sum(
                data_y[indices_inside] * weight_field[indices_inside]
            )
            if sum(weight_field[indices_inside]) == 0.0:
                hist_y[i] = hist_y[i] / 1e-10
            else:
                hist_y[i] = hist_y[i] / sum(weight_field[indices_inside])

    return np.array(hist_x), np.array(hist_y)


@dataclass
class read_data:
    file_dir: str

    def CreateDataStructure(self, fileid) -> dict:
        print(f"reading file {fileid}")
        G = Constants().G
        filename = os.path.join(self.file_dir, f"star.out.{fileid}")
        data_dict = {}
        ds = yt.load(filename)
        ad = ds.all_data()

        DM_vel = ad[("DarkMatter", "Velocities")].v
        DM_mass = ad[("DarkMatter", "Mass")].v
        DM_coord = ad[("DarkMatter", "Coordinates")].v
        Gas_coord_i = ad[("Gas", "Coordinates")].v
        Gas_density = ad[("Gas", "rho")].v
        Gas_mass = ad[("Gas", "Mass")].v
        vx = ad[("Gas", "vx")].v
        vy = ad[("Gas", "vy")].v
        vz = ad[("Gas", "vz")].v
        Gas_velocity = np.array((vx, vy, vz)).T
        Gas_temperature = ad[("Gas", "Temperature")].v
        Gas_volume = Gas_mass / Gas_density

        t = ds.current_time.in_units("day")

        # Radial and Tangential Velocity
        cm_coord = np.average(DM_coord, axis=0, weights=DM_mass)
        cm_vel = np.average(DM_vel, axis=0, weights=DM_mass)
        Gas_coord = Gas_coord_i - cm_coord
        Gas_radius = np.linalg.norm(Gas_coord, axis=-1)
        r_hat = np.divide(Gas_coord, Gas_radius[:, np.newaxis])
        vel_rel = Gas_velocity - cm_vel[np.newaxis, :]
        RadialV_vec = np.multiply(
            np.sum(r_hat * vel_rel, axis=-1)[:, np.newaxis], r_hat
        )
        Gas_RadialV = np.linalg.norm(RadialV_vec, axis=-1)
        TangentialV_vec = vel_rel - RadialV_vec
        Gas_TangentialV = np.linalg.norm(TangentialV_vec, axis=-1)
        Gas_Velocity_norm = np.linalg.norm(Gas_velocity, axis=-1)

        # Energy
        Gas_phi = Gas_mass * ad[("Gas", "Phi")].v
        Gas_ie = Gas_mass * ad[("Gas", "ie")].v / G
        Gas_ke = 0.5 * Gas_mass * Gas_RadialV**2.0
        Gas_TotalE = Gas_phi + Gas_ie + Gas_ke

        # Save in cgs, current_time in days
        data_dict["Gas_RadialV"] = Gas_RadialV * np.sqrt(G)
        data_dict["Gas_velocity"] = Gas_velocity * np.sqrt(G)
        data_dict["Gas_radius"] = Gas_radius
        data_dict["current_time"] = t  # /np.sqrt(G)
        data_dict["Gas_density"] = Gas_density
        data_dict["Gas_mass"] = Gas_mass
        data_dict["Gas_volume"] = Gas_volume
        data_dict["Gas_TangentialV"] = Gas_TangentialV * np.sqrt(G)
        data_dict["Gas_phi"] = Gas_phi * G
        data_dict["Gas_ie"] = Gas_ie * G
        data_dict["Gas_ke"] = Gas_ke * G
        data_dict["Gas_TotalE"] = Gas_TotalE * G
        data_dict["Gas_temperature"] = Gas_temperature
        data_dict["Gas_velocity_norm"] = Gas_Velocity_norm
        return data_dict

    def get_nstar(self, primary_radius) -> int:
        ds = yt.load(os.path.join(self.file_dir, "star.out.000001"))
        ad = ds.all_data()

        Rgiant = primary_radius * Constants().Rsun
        coord = ad[("Gas", "Coordinates")]
        r = np.linalg.norm(coord, axis=-1)
        Nstar = int(r[r < Rgiant].size)
        return Nstar

    def average_quantities(self, startid, endid, stepsize=10) -> dict:
        id_list = np.arange(int(startid), int(endid) + stepsize, stepsize)
        fileid_list = ["{0:06d}".format(num) for num in id_list]

        full_data_dict = {}
        for fileid in fileid_list:
            try:
                full_data_dict[fileid] = self.CreateDataStructure(fileid)
            except Exception:
                logging.exception(f"Failed to load DataStructure for {fileid}")
                continue

        average_data = {}
        # average_data['Gas_RadialV'] = {}
        average_data["Gas_RadialV"] = np.sum(
            np.array(
                [
                    full_data_dict[fileid]["Gas_RadialV"]
                    for fileid in full_data_dict.keys()
                ]
            ),
            axis=0,
        ) / len(full_data_dict.keys())
        average_data["Gas_velocity_norm"] = np.sum(
            np.array(
                [
                    full_data_dict[fileid]["Gas_velocity_norm"]
                    for fileid in full_data_dict.keys()
                ]
            ),
            axis=0,
        ) / len(full_data_dict.keys())
        average_data["Gas_velocity"] = np.sum(
            np.array(
                [
                    full_data_dict[fileid]["Gas_velocity"]
                    for fileid in full_data_dict.keys()
                ]
            ),
            axis=0,
        ) / len(full_data_dict.keys())
        return average_data

    def get_full_radial_profile(self, fileid, radial_bins=None, weight_filed=None) -> dict:
        data = self.CreateDataStructure(fileid)
        Density = data['Gas_density']
        Temperature = data['Gas_temperature'] 
        Velocity = data['Gas_velocity'] 
        IE = data['Gas_TotalE'] 
        Radius = data['Gas_radius']
        Volume = data['Gas_volume']

        if not radial_bins:
            radius_bins = 10.**np.arange(np.log10(min(Radius)),np.log10(max(Radius)),0.05)
        RadialValues, Temperature_hist = CreateRadialProfile(Radius, Temperature,bins_x = radius_bins)
        _, IE_hist = CreateRadialProfile(Radius, IE,bins_x = radius_bins)
        _, Density_hist = CreateRadialProfile(Radius, Density,bins_x = radius_bins)

        profile_dict = {}
        profile_dict['Radius'] = RadialValues.tolist()
        profile_dict['Temperature'] = Temperature_hist.tolist()
        profile_dict['Density'] = Density_hist.tolist()
        profile_dict['TotalIE'] = IE_hist.tolist()

        return profile_dict
