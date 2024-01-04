import os
import argparse
import matplotlib
import yt
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


parser = argparse.ArgumentParser()

parser.add_argument(
    "--rundir",
    help="path to star.out* files form yt",
    type=str,
    default="./",
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

parser.add_argument(
    "--projection-axis",
    type=str,
    help="axis for projection",
    default="z",
)

parser.add_argument(
    "--projection-param",
    type=str,
    help="parameter for projection",
    default="density",
)

parser.add_argument(
    "--nref",
    type=int,
    help="resolution of pictures(1 is highest, 64 is lowest)",
    default=32,
)
parser.add_argument(
    "--do-marks",
    action="store_true",
    help="mark positions of core and companion",
)
parser.add_argument(
    "--colormap", type=str, help="colormap for the plots", default="inferno"
)
parser.add_argument(
    "--colormap-lower",
    type=float,
    help="lower limit for colormap",
    default=6.0e4,
)
parser.add_argument(
    "--colormap-upper",
    type=float,
    help="upper limit for colormap",
    default=4.0e8,
)

args = parser.parse_args()

plt.switch_backend("agg")


nfiles = args.nfiles
first = args.first
interval = args.interval
nref = args.nref
projection_axis = args.projection_axis
projection_param = args.projection_param
do_marks = args.do_marks
colormap = args.colormap
colormap_lower = args.colormap_lower
colormap_upper = args.colormap_upper
rundir = os.path.abspath(args.rundir)
dataname = rundir + "/star.out."

params = {
    "backend": "agg",
    "axes.labelsize": 24,  # fontsize for x and y labels (was 10)
    "axes.titlesize": 24,
    "axes.labelweight": "heavy",
    "legend.fontsize": 18,  # was 10
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "text.usetex": True,
    "figure.figsize": [8, 6],
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.weight": "heavy",
    "font.size": 24,
    "lines.linewidth": 2,
}

matplotlib.rcParams.update(params)


dPeriod = 1e16
lim = dPeriod / 2.0 * 1.0001
hbox = np.array([[-lim, lim], [-lim, lim], [-lim, lim]])

for i in range(0, nfiles):
    num = first + i * interval
    x = "{0:06d}".format(num)
    fname = glob(dataname + x)
    if fname == []:
        continue

    print("reading dataset " + dataname + x)
    plt.clf()
    fig, ax = plt.subplots(1, 1)
    ds = yt.load(fname[0], n_ref=nref, bounding_box=hbox)
    time = ds.current_time
    ad = ds.all_data()
    dm_pos = ad[("DarkMatter", "Coordinates")].v
    dm_mass = ad[("DarkMatter", "Mass")].v
    core = dm_pos[0][:]
    comp = dm_pos[1][:]
    com = np.average(dm_pos[:, :], weights=dm_mass, axis=0)
    print("COM = ", com)
    plot = yt.ProjectionPlot(
        ds,
        projection_axis,
        ("gas", projection_param),
        center=com,
        width=5.0e15,
        fontsize=35,
    )
    plot.set_cmap(field="all", cmap=colormap)
    plot.set_zlim("all", colormap_lower, colormap_upper)

    timestr = str(time.in_units("day"))[0:5]
    plot.annotate_text(
        (0.02, 0.02),
        "t = " + timestr + " " + "d",
        coord_system="axis",
        text_args={"color": "white", "fontsize": 50},
    )

    if do_marks:
        ad = ds.all_data()
        dm_pos = ad[("DarkMatter", "Coordinates")]
        core = dm_pos[0][:]
        comp = dm_pos[1][:]
        plot.annotate_marker(
            core, coord_system="data", plot_args={"color": "black"}, marker="+"
        )
        plot.annotate_marker(
            comp,
            coord_system="data",
            plot_args={"color": "green", "s": 25},
            marker="x",
        )

        plot.annotate_marker(
            com, coord_system="data", plot_args={"color": "black"}, marker="+"
        )

    saveas = projection_param + "_" + projection_axis + "_" + x + ".pdf"
    plot.set_colorbar_label(
        field=("gas", projection_param),
        label="Projected " + projection_param + r" (${\rm g\,cm}^{-2}$)",
    )
    plot.save(saveas)
    print("saved figure " + saveas)
    plt.tight_layout()
    plt.clf()
