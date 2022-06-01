from .classes import FitsImage
from .utils import is_within, plot_2D_map
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from astropy.visualization import ImageNormalize
from matplotlib import ticker
import matplotlib.patheffects as pe

def get_figsize(ncols, nrows, max_size=()):
    return (ncols * 3, nrows * 3)

def get_imagegrid(npanel, pad=0.0):
    ncols = np.ceil(npanel**0.5).astype(int)
    nrows = np.ceil(npanel / ncols).astype(int)
    fig = plt.figure(figsize=get_figsize(ncols, nrows))
    imgrid = ImageGrid(
        fig, rect=111, nrows_ncols=(nrows, ncols), share_all=True, axes_pad=pad
    )
    return fig, imgrid

def left_bottom_ax(imgrid):
    nrows, ncols = imgrid.get_geometry()
    i = ncols * (nrows - 1)
    return imgrid[i]

def plot_channel_map(
    fitsname,
    center_coord=None,
    rmax=10.0,
    vrange=None,
    thin=1,
    sigma_clip=None,
    rms=None,
    noisemask_kw=dict(rmin=10.0, rmax=15.0),
    pad=0.0,
    cmap_kw=dict(),
):
    # load the imagecube
    print("Loading data...")
    imagecube = FitsImage(fitsname)
    imagecube.get_directional_coord(center_coord=center_coord)
    imagecube.get_spectral_coord()

    # measure the rms for better visualization
    if rms is None and sigma_clip is not None:
        print("Estimating rms...")
        imagecube.estimate_rms(**noisemask_kw)
        rms = imagecube.rms
        print("rms: {:.2f} mJy/beam".format(imagecube.rms * 1e3))

    # select the velocity channels to plot
    velax = imagecube.v
    if vrange is not None:
        velax = velax[is_within(imagecube.v, vrange)]
    velax = velax[::thin]  # skip each *thin* channel to reduce the number of channels to plot

    # setup imagegrid
    fig, imgrid = get_imagegrid(velax.size, pad=pad)

    # image normalization and clipping
    data = imagecube.data
    if vrange is not None:
        data = data[is_within(imagecube.v, vrange),:,:] # limit to relevant channels
    data = data[::thin] # skip each *thin* channel to reduce the number of channels to plot

    if sigma_clip is not None:
        data[data < sigma_clip * rms] = np.nan
    norm = ImageNormalize(
        data, vmin=sigma_clip * rms if sigma_clip is not None else 0.0
    )

    cmap_kw["norm"] = norm

    # iterate over channels to plot
    for i, v in enumerate(velax):
        # define ax
        ax = imgrid[i]

        # get data
        # im = data[imagecube.v == v, :, :].squeeze()
        im = data[i]

        # plot
        print("Plotting v = {:.2f} km/s...".format(v))
        plot_2D_map(
            data=im,
            X=imagecube.x,
            Y=imagecube.y,
            ax=ax,
            cmap=True,
            cmap_method="pcolorfast",
            contour=False,
            colorbar=False,
            cmap_kw=cmap_kw,
        )

        # set the spatial range
        ax.set(xlim=(rmax, -rmax), ylim=(-rmax, rmax))

        # annotate the velocity value of the channel
        ax.annotate(
            text="{:.2f} km/s".format(v),
            xy=(0.95, 0.95),
            ha="right",
            va="top",
            xycoords="axes fraction",
            color="black",
            path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()],
        )

        # set ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    ### ImageGrid should have param of *ngrids*, but specifying this param cause an error (maybe bug?).
    ### Here is workaround for that, removing axes on which no data are drawn.
    for i in range(i + 1, len(imgrid)):
        imgrid[i].set_axis_off()

    # axes label in the bottom left panel
    left_bottom_ax(imgrid).set(
        xlabel="$\Delta$R.A. [arcsec]", ylabel="$\Delta$Dec. [arcsec]"
    )

    return fig