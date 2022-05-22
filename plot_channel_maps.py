import numpy as np
from qdisk.classes import FitsImage
from astropy.coordinates import SkyCoord
from qdisk.utils import plot_2D_map
import matplotlib.pyplot as plt
from eDisk_source_dict import source_dict
from mpl_toolkits.axes_grid1 import ImageGrid, Grid
from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib import ticker
from eDiskplot import cmap
import matplotlib.patheffects as pe 
import sys

plt.rcParams.update({
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in"})

imagepath = "/raid/work/yamato/edisk_data/L1489IRS/eDisk_image_products/"
source = sys.argv[1]
molecular_lines = sys.argv[2]
baseline = sys.argv[3]
# vrange = source_dict[source]["emission_extent"][molecular_lines]
vrange = (5.8, 8)
rmax = 5 #x.5 is recommended for nicer ticker
sigma_clip = 2
center_coord = SkyCoord(source_dict[source]["radec"], frame="icrs")

# load the image cube and coordinate
imagename = imagepath + "{:s}_{:s}_{:s}_robust_0.5.image.fits".format(
    source, baseline, molecular_lines
)
imagecube = FitsImage(imagename)
imagecube.get_directional_coord(center_coord=center_coord)
imagecube.get_spectral_coord()
print("Estimating rms...")
imagecube.estimate_rms(rmin=12, rmax=15)
print("rms: {:.2f} mJy/beam".format(imagecube.rms*1e3))

# select channels to plot
velax = imagecube.v[(imagecube.v >= vrange[0]) & (imagecube.v <= vrange[1])]

# set up image grid
ncols = np.ceil(velax.size**0.5).astype(int)
nrows = np.ceil(velax.size / ncols).astype(int)
# fig, axes = plt.subplots(ncols=ncols, nrows=nrows, sharex=True, sharey=True)
fig = plt.figure(figsize=(ncols*3, nrows*3))
imgrid = ImageGrid(fig, rect=111, nrows_ncols=(nrows, ncols), share_all=True, axes_pad=0.0)

def left_bottom_ax(imgrid, nrows, ncols):
    i = ncols * (nrows - 1)
    return imgrid[i]

for i, v in enumerate(velax):
    ax = imgrid[i]
    data = imagecube.data[imagecube.v == v, :, :].squeeze()
    data[data < sigma_clip*imagecube.rms] = np.nan
    norm = ImageNormalize(vmin=sigma_clip*imagecube.rms)

    print("Plotting v = {:.2f} km/s...".format(v))
    plot_2D_map(
        data=data,
        X=imagecube.x,
        Y=imagecube.y,
        ax=ax,
        cmap=True,
        cmap_method="pcolorfast",
        contour=False,
        colorbar=False,
        cmap_kw=dict(cmap=cmap["channel_map"], norm=norm),
    )
    ax.set(xlim=(rmax, -rmax), ylim=(-rmax, rmax))
    ax.annotate(text="{:.2f} km/s".format(v), xy=(0.95, 0.95), ha="right", va="top", xycoords="axes fraction", color="black", path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

### ImageGrid should have param of *ngrids*, but specifying this param cause an error (maybe bug?).
### Here is workaround for that, removing axes on which no data are drawn.
for i in range(i+1, len(imgrid)):
    imgrid[i].set_axis_off()

left_bottom_ax(imgrid, nrows, ncols).set(xlabel="$\Delta$R.A. [arcsec]", ylabel="$\Delta$Dec. [arcsec]")
# imgrid.set_axes_pad((0.0, 0.0))

# fig.savefig("./test.pdf", dpi=300, bbox_inches="tight")
plt.show()
