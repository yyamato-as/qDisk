from multiprocessing.sharedctypes import Value
import numpy as np
import astropy.io.fits as fits
import qdisk.utils as utils
from qdisk.classes import FitsImage
import bettermoments as bm
from scipy.interpolate import griddata

moment_method = {
    0: "zeroth",
    1: "first",
    8: "eighth",
}

deg_to_arcsec = 3600.

def process_chunk_array(func, args, targname, nchunks=4, axis=0, concataxis=0, verbose=True):
    """Processing the large array which is out-of-memory using chunking. Assume the function will return two arrays.

    Parameters
    ----------
    func : function
        The function you want to calculate
    args : dict
        A dictionary of arguments for ``func``
    targname : str
        The name of argument you want to chunk
    nchunks : int, optional
        Number of chunks, by default 4
    axis : int, optional
        The axis over which the data will be splitted, by default 0

    Returns
    -------
    tuple of two arrays
        restored arrays
    """

    split_arrays = np.split(args[targname], indices_or_sections=nchunks, axis=axis)
    prod =[]
    dprod = []
    for i, array in enumerate(split_arrays):
        args[targname] = array
        if verbose:
            print("Computing chunk {}...".format(i))
        p, dp = func(**args) # assume the func returns the two arrays; moment map and uncertainty map
        prod.append(p)
        dprod.append(dp)
    if verbose:
        print("Restoring the original array...")
    return np.concatenate(prod, axis=concataxis), np.concatenate(dprod, axis=concataxis)

def calculate_moment(
    imagename,
    moments=[0],
    noisechannel=3,
    verbose=False,
    mask=None,
    channel=None,
    vel_extent=None,
    threshold=None,
    rms=None,
    save=True,
    savefilename=None,
    nchunks=None
):
    """Compute the moment of the image using bettermoments.

    Parameters
    ----------
    imagename : str
        Path to the image cube for which you will calculate the moment.
    moments : list, optinal
        List of the moment you want to calculate in integer, by default [0]. See the method dictionary above.
    noisechannel : int, optional
        Number of channels from the both edge of the image for noise calculation, by default 3.
    verbose : bool, optional
        If the calculated rms is printed, by default False
    mask : array, optional
        Pre-defined mask, need to be in the same shape as image cube, by default None, i.e., no pre-defined mask.
    channel : str, optional
        Specification of the channels included for the moment calculation, by default None, i.e., use all channels. The same specification manner as CASA ``chans`` parameter.
    vel_extent : tuple, optional
        A tuple which contains the minimum and maximum velocity for moment calculation, by default None, i.e., use all velocity range of the image cube
    threshold : tuple, optional
        A tuple which contains the minimum and maximum value for clipping in sigma, by default None, i.e., no sigma clipping. (e.g., (-5, 3) means exclude the pixels which have values in between -5sigma to 3sigma)
    rms : float, optional
        Manual specification of rms, by default None, i.e., estimate rms by bettermoments method ``estimate_RMS``.
    save : bool, optional
        If the calculated moment is saved into a FITS, by default True.
    savefilename : str, optional
        Path of the saved FITS, by default None, i.e., the saved file will be named as imagepath.replace(".fits", "_M0.fits") etc.

    Returns
    -------
    moments (tuple)
        A tuple which contains the moment map array and its uncertainty array.
    """

    print("Loading data...")
    data, velax = bm.load_cube(imagename)  # velax in m/s

    if verbose:
        print("data shape: {}".format(data.shape))

    print("Estimating rms...")
    rms = bm.estimate_RMS(data=data, N=noisechannel) if rms is None else rms

    if verbose:
        print("rms: {} mJy/beam".format(rms * 1e3))

    # user mask
    mask = np.ones(data.shape) if mask is None else mask

    # threshold mask
    print("Generating threshold mask...")
    if threshold is not None:
        tmask = bm.get_threshold_mask(data=data, clip=threshold)
    else:
        tmask = np.ones(data.shape)

    # channel mask
    if channel is None and vel_extent is None:
        print("Generating channel mask...")
        cmask = np.ones(data.shape)
    elif channel is not None:
        print("Generating channel mask based on specified channels...")
        cmask = np.zeros(data.shape)
        for cr in channel.split(";"):
            firstchannel, lastchannel = [int(c) for c in cr.split("~")]
            cmask += bm.get_channel_mask(
                data=data, firstchannel=firstchannel, lastchannel=lastchannel
            )
        cmask = np.where(cmask != 0.0, 1.0, 0.0)  # manage possible overlaps
    else:
        print("Generating channel mask based on specified velocity range...")
        firstchannel, lastchannel = [
            np.argmin(np.abs(velax - v * 1e3)) for v in vel_extent # note velax in m/s, vel_extent in km/s
        ]
        cmask = bm.get_channel_mask(
            data=data, firstchannel=firstchannel, lastchannel=lastchannel
        )

    # mask combination
    print("Combining the mask...")
    mask = bm.get_combined_mask(
        user_mask=mask, threshold_mask=tmask, channel_mask=cmask, combine="and"
    )

    # masked data
    data *= mask

    # moment calc by bettermoments; moment 1 may take time
    maps = {}
    for mom in moments:
        print("Calculating moment {}...".format(mom))
        collapse = getattr(bm, "collapse_{:s}".format(moment_method[mom]))
        if nchunks is not None:
            print("Going to compute with {} chunks...".format(nchunks))
            args = {"velax": velax, "data": data, "rms": rms}
            M = process_chunk_array(collapse, args, "data", nchunks=nchunks, axis=2, concataxis=1)
        else:
            M = collapse(velax=velax, data=data, rms=rms)
        if save:
            bm.save_to_FITS(moments=M, method=moment_method[mom], path=imagename, outname=savefilename)
        maps[mom] = M

    return maps


def calculate_radial_profile(imagepath, PA=0., incl=45., center_coord=None, rbins=None, rmin=0.0, rmax=None, wedge_angle=30, mask=None, assume_correlated=False):

    print("Loading data...")
    im = FitsImage(imagepath, bpa_from_west=True)
    im.get_projected_coord(PA=PA, incl=incl, center_coord=center_coord)

    if im.ndim != 2:
        raise ValueError("The data is not in 2D. Radial profile calculation failed.")

    if rbins is None:
        rbin_width = im.bmaj / 4.0 # 1/4 of bmaj in arcsec
        if rmax is None:
            rmax = np.nanmax(im.r)
        rbins = np.arange(rmin, rmax, rbin_width)
    
    rvals = np.average([rbins[1:], rbins[:-1]], axis=0)

    theta_exclude = ((im.theta > -180. + 0.5*wedge_angle) & (im.theta < -0.5*wedge_angle)) | ((im.theta > 0.5*wedge_angle) & (im.theta < 180. - 0.5*wedge_angle))
    theta_mask = np.logical_not(theta_exclude)

    if mask is not None:
        mask *= theta_mask
    else:
        mask = theta_mask

    mask = mask.flatten()
    rpnts = im.r.flatten()[mask]
    toavg = im.data.flatten()[mask]
    ridxs = np.digitize(rpnts, rbins)

    # calculate number of beams per bin
    if assume_correlated:
        nbeams = np.array([im.Omega_pix * len(toavg[ridxs == r]) / im.Omega_beam for r in range(1, rbins.size)]) 
    else:
        nbeams = 1
    
    print("Calculating radial profile...")
    ravgs = np.array([np.mean(toavg[ridxs == r]) for r in range(1, rbins.size)])
    rstds = np.array([np.std(toavg[ridxs == r]) for r in range(1, rbins.size)])
    rstds /= np.sqrt(nbeams)

    return rvals, ravgs, rstds


def calculate_averaged_spectra(imagename, **mask_kwargs):

    image = FitsImage(imagename)
    image.get_spectral_coord()
    image.get_mask(**mask_kwargs)

    masked_data = image.data * image.mask

    avgspec = np.array([np.nanmean(im[im.nonzero()]) for im in masked_data])

    return image.v, avgspec


def calculate_pvdiagram(imagename, center_coord=None, PA=90., incl=0.0, rrange=(-10.0, 10.0), vrange=None):
    # fetch FitsImage class
    image = FitsImage(imagename)

    # construct original axes
    image.get_projected_coord(center_coord=center_coord, PA=PA, incl=incl)
    image.get_spectral_coord()

    # construct interpolate axes
    x_ip = np.arange(*rrange, abs(image.dx)*deg_to_arcsec)
    y_ip = np.zeros_like(x_ip) # trace y = 0 on the projected coord

    # masking; as the data is huge, it is important to limit the region for calculation by applying a nominal mask to reduce the calculation time
    x_pad = image.bmaj
    y_pad = image.bmaj / np.cos(np.radians(incl))
    x_mask = (image.x_proj >= rrange[0] - x_pad) & (image.x_proj <= rrange[1] + x_pad)
    y_mask = (image.y_proj >= -y_pad) & (image.y_proj <= y_pad)

    # chennel selection
    if vrange is None:
        data = image.data
        velax = image.v
    else:
        v_mask = (image.v >= vrange[0]) & (image.v <= vrange[1])
        data = image.data[v_mask, :, :] # exclude non-relevant channels
        velax = image.v[v_mask]

    # mask out spatially
    data[:, ~y_mask | ~x_mask] = 0.0 # put zeros into non-relevant pixels
    x_orig = image.x_proj[y_mask & x_mask].flatten()
    y_orig = image.y_proj[y_mask & x_mask].flatten()

    # interpolate over each channel
    pvdiagram = np.array([griddata(np.array([y_orig, x_orig]).T, im[im.nonzero()].flatten(), (y_ip, x_ip), method='cubic') for im in data])

    return x_ip, velax, pvdiagram
        




