import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from cee_profile import module

parser = argparse.ArgumentParser()

parser.add_argument(
    "--rundir",
    help="path to star.out* files form yt",
    type=str,
    default="./",
)
parser.add_argument(
    "--primary-radius",
    help="radius of the redgiant in unit Rsun",
    type=float,
    default=52,
)
parser.add_argument(
    "--nfiles",
    type=int,
    help="number of output files to analyze",
    default=1000,
)
parser.add_argument("--first", type=int, help="start at this file", default=1)
parser.add_argument(
    "--interval", type=int, help="interval between output files", default=5
)


args = parser.parse_args()

plt.switch_backend("agg")


nfiles = args.nfiles
first = args.first
interval = args.interval

rundir = os.path.abspath(args.rundir)
dataname = rundir + "/star.out."


Nstar = module.read_data(rundir).get_nstar(args.primary_radius)

with open("average_data.txt", "w") as ff:
    ff.write(
        "# Time(day) AverageDensity(cgs) AverageDensity_VolWgt(cgs)"
        " AverageDensity_unb(cgs) AverageDensity_VolWgt_unb(cgs)"
    )
    ff.write("\n")
for i in range(0, nfiles):
    num = first + i * interval
    x = "{0:06d}".format(num)
    fname = glob(dataname + x)
    if fname == []:
        continue
    data = module.read_data(rundir).CreateDataStructure(x)
    Gas_radius = data["Gas_radius"][:Nstar]
    Gas_RadialV = data["Gas_RadialV"][:Nstar]
    Gas_TangentialV = data["Gas_TangentialV"][:Nstar]
    Gas_density = data["Gas_density"][:Nstar]
    Gas_TotalE = data["Gas_TotalE"][:Nstar]
    Gas_mass = data["Gas_mass"][:Nstar] / 1e3
    Gas_volume = data["Gas_volume"][:Nstar]

    AverageDensity = np.average(Gas_density)
    AverageDensity_VolWgt = np.average(Gas_density, weights=Gas_volume)
    Time = data["current_time"].v

    inds = np.where(Gas_TotalE > 0.0)[0]

    AverageDensity_unb = np.average(Gas_density[inds])
    AverageDensity_VolWgt_unb = np.average(
        Gas_density[inds], weights=Gas_volume[inds]
    )
    with open("average_data.txt", "a") as ff:
        line = (
            f"{Time} {AverageDensity} {AverageDensity_VolWgt} "
            f"{AverageDensity_unb} {AverageDensity_VolWgt_unb}"
        )
        ff.write(line)
        ff.write("\n")
