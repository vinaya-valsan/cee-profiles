import os
import sys
import yt
import numpy as np
from dataclasses import dataclass


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

        ## Radial and Tangential Velocity
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

    def initial_rotation(self) -> float:
        ds = yt.load(os.path.join(self.file_dir, "star.out.000001"))
        ad = ds.all_data()

        # RG core
        RG_core_vel_vec = ad[("DarkMatter", "Velocities")][0].v
        RG_core_mass = ad[("DarkMatter", "Mass")][0].v
        RG_core_cord_vec = ad[("DarkMatter", "Coordinates")][0].v

        # companion
        Comp_core_vel_vec = ad[("DarkMatter", "Velocities")][1].v
        Comp_core_mass = ad[("DarkMatter", "Mass")][1].v
        Comp_core_cord_vec = ad[("DarkMatter", "Coordinates")][1].v

        Comp_core_rel_cord_vec = (
            Comp_core_cord_vec - RG_core_cord_vec
        )  # relative coordinate of companion wrt RG core
        Comp_core_radius = np.linalg.norm(
            Comp_core_rel_cord_vec
        )  # Radius of companion from RG core

        r_hat = (
            Comp_core_rel_cord_vec / Comp_core_radius
        )  # unit vector in the radial direction from RG  core to companion

        Comp_core_rel_velocity_vec = (
            Comp_core_vel_vec - RG_core_vel_vec
        )  # Relative velocity of Companion wrt to RG core
        Comp_core_radial_velocity_vec = r_hat * np.dot(
            r_hat, Comp_core_rel_velocity_vec
        )  # Radial velocity vector of Companion wrt RG core
        Comp_core_radial_velocity = np.linalg.norm(
            Comp_core_radial_velocity_vec
        )  # Radial velocity of companion wrt RG core
        Comp_core_tangential_velocity_vec = (
            Comp_core_rel_velocity_vec - Comp_core_radial_velocity_vec
        )  # Tangential velocity vector of Companion wrt RG core
        Comp_core_tangential_velocity = np.linalg.norm(
            Comp_core_tangential_velocity_vec
        )  # Tangential velocity of Companion wrt RG core

        omega = (
            Comp_core_tangential_velocity / Comp_core_radius
        )  # orbital frequency

        # star particles
        Gas_coord_i = ad[
            ("Gas", "Coordinates")
        ].v  # coordinates of star particles
        vx = ad[("Gas", "vx")].v
        vy = ad[("Gas", "vy")].v
        vz = ad[("Gas", "vz")].v
        Gas_velocity = np.array((vx, vy, vz)).T  # velocity of star particles

        Gas_coord = (
            Gas_coord_i - RG_core_vel_vec
        )  # relative coordinates star particles wrt RG core
        Gas_radius = np.linalg.norm(
            Gas_coord, axis=-1
        )  # Radius of star particles from RG core
        r_hat = np.divide(
            Gas_coord, Gas_radius[:, np.newaxis]
        )  # radial unit vector
        vel_rel = (
            Gas_velocity - RG_core_vel_vec[np.newaxis, :]
        )  # relative velocity of star particles wrt RG core

        RadialV_vec = np.multiply(
            np.sum(r_hat * vel_rel, axis=-1)[:, np.newaxis], r_hat
        )  # Relative radial velocity vector of star particles wrt RG core
        Gas_RadialV = np.linalg.norm(
            RadialV_vec, axis=-1
        )  # Relative radial velocity of star particles wrt RG core
        TangentialV_vec = (
            vel_rel - RadialV_vec
        )  # tangential velocty of star particles wrt RG core
        Gas_TangentialV = np.linalg.norm(TangentialV_vec, axis=-1)

        fr = Gas_TangentialV / (Gas_radius * omega)
        self.fr = fr
        self.omega = omega

    def average_quantities(self, startid, endid, stepsize=10) -> dict:
        id_list = np.arange(int(startid), int(endid) + stepsize, stepsize)
        fileid_list = ["{0:06d}".format(num) for num in id_list]

        full_data_dict = {}
        for fileid in fileid_list:
            try:
                full_data_dict[fileid] = self.CreateDataStructure(fileid)
            except:
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
