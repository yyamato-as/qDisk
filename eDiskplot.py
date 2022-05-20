import seaborn as sns
import qdisk.utils as utils
import astropy.io.fits as fits
from astropy.visualization import ImageNormalize
from qdisk.classes import FitsImage
import numpy as np
import scipy.constants as sc

### color pallet
from matplotlib import colors
freeze = np.loadtxt('/home/yamato/Project/MAPS/script/MAPS_cmap.txt')
freeze /= 255.0
cpal = colors.ListedColormap(freeze, name='freeze')

cmap = {"continuum": "inferno", "M0": cpal, "M1": "RdBu_r", "M8": "gist_heat"}


def plot_map(
    fitsname,
    ax,
    center_coord=None,
    data_scaling_factor=1.0,
    in_Tb=False,
    scale=50,
    distance=140,
    cmap=None,
    colorbar=True,
    norm_kwargs=dict(),
    cbar_kw=dict(),
    beam_kw=dict(),
    sbar_kw=dict()
):

    # read FITS
    image = FitsImage(fitsname)
    data = image.data * data_scaling_factor

    # any conversion
    if in_Tb:
        data = utils.jypb_to_K(data, image.restfreq, image.beam[:2])

    if image.ndim != 2:
        raise ValueError("The data is not in 2D.")

    # generate directional axis
    image.get_directional_coord(center_coord=center_coord)

    # scale bar
    scale = (scale / distance, "{} au".format(int(scale)))

    # normalization
    norm = ImageNormalize(data, **norm_kwargs)

    # plot
    utils.plot_2D_map(
        data=data,
        X=image.x,
        Y=image.y,
        ax=ax,
        cmap=True,
        cmap_method="pcolorfast",
        contour=False,
        colorbar=colorbar,
        beam=image.beam,
        scale=scale,
        cmap_kw=dict(cmap=cmap, norm=norm),
        cbar_kw=cbar_kw,
        beam_kw=beam_kw,
        sbar_kw=sbar_kw
    )

def overlay_contour(
    fitsname, 
    ax, 
    center_coord=None, 
    data_scaling_factor=1.0,
    levels=None,
    colors="black"
    ):

    image = FitsImage(fitsname)
    data = image.data * data_scaling_factor

    image.get_directional_coord(center_coord=center_coord)

    utils.plot_2D_map(
        data=data,
        X=image.x,
        Y=image.y,
        ax=ax,
        cmap=False,
        contour=True,
        colorbar=False,
        contour_kw=dict(levels=levels, colors=colors)
    )


def plot_radial_profile(filename, ax, color="tab:blue", normalize=False, scale=1.0, label=None):

    x, y, dy = np.loadtxt(filename, unpack=True)

    if normalize:
        dy /= y.max()
        y /= y.max()

    y *= scale
    dy *= scale

    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, y-dy, y+dy, facecolor=color, alpha=0.25, edgecolor=None)
    # ax.axhline(y=0.0, color="gray", ls="dashed")

    ax.set_aspect(1./ax.get_data_ratio()/sc.golden_ratio)
    ax.set(xlim=(x.min(), x.max()), ylim=(-dy.mean(), None))


