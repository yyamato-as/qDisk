import seaborn as sns
import utils
import astropy.io.fits as fits
from astropy.visualization import ImageNormalize
from matplotlib.colors import Colormap 
import numpy as np

### color pallet
from matplotlib import colors
freeze = np.loadtxt('/home/yamato/Project/MAPS/script/MAPS_cmap.txt')
freeze /= 255.0
cpal = colors.ListedColormap(freeze, name='freeze')


def plot_map(
    fitsname,
    ax,
    center_coord=None,
    scale=50,
    distance=140,
    unit=r"mJy beam$^{-1}$",
    cmap=None,
    norm_kwargs=dict(),
):

    # read FITS
    header = fits.getheader(fitsname)
    data = fits.getdata(fitsname).squeeze()

    if data.ndim != 2:
        raise ValueError("The data is not in 2D.")

    # generate directional axis
    x, y = utils.get_radec_coord(header, center_coord=center_coord)

    # get beam data and scale
    beam = utils.fetch_beam_info(header, pa_rotate=True)
    scale = (scale / distance, "{} au".format(int(scale)))

    # normalization
    norm = ImageNormalize(data, **norm_kwargs)

    # plot
    utils.plot_2D_map(
        data=data,
        X=x,
        Y=y,
        ax=ax,
        cmap=True,
        cmap_method="pcolorfast",
        contour=False,
        colorbar=True,
        beam=beam,
        scale=scale,
        cmap_kw=dict(cmap=cmap, norm=norm),
        cbar_kw=dict(label=unit)
    )
