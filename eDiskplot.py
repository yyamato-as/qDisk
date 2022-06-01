import seaborn as sns
import qdisk.utils as utils
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from astropy.visualization import ImageNormalize
from qdisk.classes import FitsImage, PVFitsImage
from matplotlib import ticker
import matplotlib.patheffects as pe
import numpy as np
import scipy.constants as sc

### color pallet
from matplotlib import colors

freeze = np.loadtxt("/home/yamato/Project/qDisk/cmap_freeze.txt")
freeze /= 255.0
cpal = colors.ListedColormap(freeze, name="freeze")

cmap = {
    "continuum": "inferno",
    "M0": cpal,
    "M1": "RdBu_r",
    "M8": "gist_heat",
    "channel_map": cpal,
}


def is_within(value, range):
    return np.logical_and(value >= range[0], value <= range[1])


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


def plot_PV_diagram(
    fitsname,
    ax,
    data_scaling_factor=1,
    xaxis="pos",
    cmap=None,
    colorbar=True,
    norm_kwargs=dict(),
    cbar_kw=dict(),
    beam_kw=dict(),
    sbar_kw=dict(),
):

    # read FITS
    image = PVFitsImage(fitsname)

    if image.ndim != 2:
        raise ValueError("The data is not in 2D.")

    # data
    data = image.data * data_scaling_factor

    # get coordinate
    image.get_coord()

    # normalization
    norm = ImageNormalize(data, **norm_kwargs)

    # plot
    utils.plot_2D_map(
        data=data,
        X=image.p if xaxis == "pos" else image.v,
        Y=image.v if xaxis == "pos" else image.p,
        ax=ax,
        cmap=True,
        cmap_method="pcolorfast",
        contour=False,
        colorbar=colorbar,
        cmap_kw=dict(cmap=cmap, norm=norm),
        cbar_kw=cbar_kw,
        beam_kw=beam_kw,
        sbar_kw=sbar_kw,
    )


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
    velax = imagecube.v[is_within(imagecube.v, vrange)]
    velax = velax[::thin]  # skip each *thin* channels to reduce the number of channels to plot

    # setup imagegrid
    fig, imgrid = get_imagegrid(velax.size, pad=pad)

    # image normalization and clipping
    data = imagecube.data[is_within(imagecube.v, vrange),:,:][::thin] # limit to relevant channels
    if sigma_clip is not None:
        data[data < sigma_clip * rms] = np.nan
    norm = ImageNormalize(
        data, vmin=sigma_clip * rms if sigma_clip is not None else 0.0
    )

    # iterate over channels to plot
    for i, v in enumerate(velax):
        # define ax
        ax = imgrid[i]

        # get data
        # im = data[imagecube.v == v, :, :].squeeze()
        im = data[i]

        # plot
        print("Plotting v = {:.2f} km/s...".format(v))
        utils.plot_2D_map(
            data=im,
            X=imagecube.x,
            Y=imagecube.y,
            ax=ax,
            cmap=True,
            cmap_method="pcolorfast",
            contour=False,
            colorbar=False,
            cmap_kw=dict(cmap=cmap["channel_map"], norm=norm),
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
    sbar_kw=dict(),
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
        sbar_kw=sbar_kw,
    )


def overlay_contour(
    fitsname,
    ax,
    center_coord=None,
    data_scaling_factor=1.0,
    levels=None,
    colors="black",
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
        contour_kw=dict(levels=levels, colors=colors),
    )


def plot_radial_profile(
    filename, ax, color="tab:blue", normalize=False, scale=1.0, label=None
):

    x, y, dy = np.loadtxt(filename, unpack=True)

    if normalize:
        dy /= y.max()
        y /= y.max()

    y *= scale
    dy *= scale

    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, y - dy, y + dy, facecolor=color, alpha=0.25, edgecolor=None)
    # ax.axhline(y=0.0, color="gray", ls="dashed")

    ax.set_aspect(1.0 / ax.get_data_ratio() / sc.golden_ratio)
    ax.set(xlim=(x.min(), x.max()), ylim=(-dy.mean(), None))
