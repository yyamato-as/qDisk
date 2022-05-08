from numpy import source
from qdisk import data_product as dp
from eDisk_source_dict import source_dict
from astropy.coordinates import SkyCoord
from astropy.visualization import AsinhStretch
import numpy as np
import matplotlib.pyplot as plt
import os
from eDiskplot import plot_map, cpal

### compare the moment 0 with different voxel selections

source = "L1489IRS"
ms = "C18O"
imagepath = "/raid/work/yamato/eDisk_data/L1489IRS/ALMA_pipeline_reduced_data/try1_continuum_nterms1/{:s}_SBLB_{:s}_robust_0.5.image.fits".format(
    source, ms
)
outfile = "/raid/work/yamato/eDisk_data/L1489IRS/data_product_test/{:s}_SBLB_{:s}_robust_0.5.image.fits".format(
    source, ms
)

r = 5  # in arcsec

moment = 0
methods = ["noselection", "vrange", "3sigmaclip", "3sigmaclip_noneg"]
kwargs = {
    "noselection": dict(),
    "vrange": dict(vel_extent=source_dict[source]["emission_extent"][ms]),
    "3sigmaclip": dict(threshold=(-3, 3)),
    "3sigmaclip_noneg": dict(threshold=(-np.inf, 3)),
}
title = {"noselection": "Whole cube",
         "vrange": "{:.1f} km/s - {:.1f} km/s".format(*[v for v in source_dict[source]["emission_extent"][ms]]),
         "3sigmaclip": r"3$\sigma$ clip (incl. absorption)",
         "3sigmaclip_noneg": r"3$\sigma$ clip (excl. absorption)"}

# for method in methods:
#     dp.calculate_moment(
#         imagepath=imagepath, moments=[moment], verbose=True, nchunks=4, save=True, savefilepath=outfile, **kwargs[method]
#     )
#     for pre in ["_M0", "_dM0"]:
#         cmd = "mv " + outfile.replace(".fits", "{:s}.fits".format(pre)) + " " + outfile.replace(".fits", "{:s}_{:s}.fits".format(pre, method))
#         os.system(cmd)

fig, axes = plt.subplots(1, len(methods), sharex=True, sharey=True, figsize=(3*len(methods), 3))

for i, (method, ax) in enumerate(zip(methods, axes)):
    fitsname = outfile.replace(".fits", "_M0_{:s}.fits".format(method))
    unit = "mJy/beam km/s" if i == len(methods)-1 else None
    plot_map(
        fitsname=fitsname,
        ax=ax,
        center_coord=SkyCoord(source_dict[source]["radec"], frame="icrs"),
        scale=50,
        distance=source_dict[source]["distance"],
        unit=unit,
        cmap=cpal,
        norm_kwargs=dict(vmin=0.0, vmax=40, stretch=AsinhStretch(a=0.1))
    )
    ax.set(xlim=(r, -r), ylim=(-r, r), title=title[method])

fig.savefig("./figure/mom0_selection_comparison_asinhstretch.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
