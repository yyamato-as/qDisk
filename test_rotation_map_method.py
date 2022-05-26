from qdisk.product import calculate_moment, moment_method
import matplotlib.pyplot as plt
from eDisk_source_dict import source_dict
from astropy.visualization import AsinhStretch, ImageNormalize
import numpy as np


plt.rcParams.update({
    "xtick.top": True,
    "ytick.right": True})

source = "L1489IRS"
molecular_lines = ["12CO", "13CO", "C18O", "SO"]
imagepath = "/raid/work/yamato/edisk_data/L1489IRS/eDisk_image_products/"
rmax = 8
rmax_zoomed = 2.5
moment = ["M0", "M1", "M8"]
baseline = "SBLB"
vsys = source_dict[source]["v_sys"]
center_coord = source_dict[source]["radec"]
vrange = 5
data_scaling_factor = {"M0": 1.0,
                       "M1": 1e-3,
                       "M8": 1,}
norm_kwargs = {"M0": dict(stretch=AsinhStretch(a=0.1), vmin=0.0),
               "M1": dict(vmin=vsys-vrange, vmax=vsys+vrange),
               "M8": dict()}
beam_kw = {"M0": dict(fill=True, color="white"),
           "M1": dict(fill=True, color="black"),
           "M8": dict(fill=True, color="white")}
cbar_kw = {"M0": dict(label=r"mJy beam$^{-1}$ km s$^{-1}$"),
           "M1": dict(label=r"km s$^{-1}$"),
           "M8": dict(label=r"K")}
sbar_kw = {"M0": dict(),
           "M1": dict(color="black"),
           "M8": dict(),
}

def get_image_basename(source, baseline, line, robust=0.5, pbcor=False):
    ### get basename of an image based on eDisk naming convention
    imagename = "{:s}_{:s}_{:s}_robust_{:.1f}.image.fits".format(source, baseline, line, robust)
    if pbcor:
        imagename = imagename.replace(".image.fits", ".pbcor.fits")
    return imagename



line = "C18O"
imagename = imagepath + get_image_basename(source, baseline, line)


for mom in [1, 9, "q"]:
    threshold = (-np.inf, 4) if mom == 1 else None
    calculate_moment(
        imagename=imagename,
        moments=[mom],
        verbose=True,
        threshold=threshold,
        savefilename="./test_rotation_map_{:s}.fits".format(moment_method[mom]),
        nchunks=4
    )