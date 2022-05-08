import numpy as np
import bettermoments as bm

moment_method = {
    0: "zeroth",
    1: "first",
    8: "eighth",
}

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


def get_corresp_channels(axis, val):
    return np.argmin(np.abs(axis - val))


def calculate_moment(
    imagepath,
    moments=[0],
    noisechannel=3,
    verbose=False,
    mask=None,
    channel=None,
    vel_extent=None,
    threshold=None,
    rms=None,
    save=True,
    savefilepath=None,
    nchunks=None
):
    """Compute the moment of the image using bettermoments.

    Parameters
    ----------
    imagepath : str
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
    savefilepath : str, optional
        Path of the saved FITS, by default None, i.e., the saved file will be named as imagepath.replace(".fits", "_M0.fits") etc.

    Returns
    -------
    moments (tuple)
        A tuple which contains the moment map array and its uncertainty array.
    """

    print("Loading data...")
    data, velax = bm.load_cube(imagepath)  # velax in m/s

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
            get_corresp_channels(velax, v * 1e3) for v in vel_extent # note velax in m/s, vel_extent in km/s
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
            bm.save_to_FITS(moments=M, method=moment_method[mom], path=imagepath, outname=savefilepath)
        maps[mom] = M

    return maps


# def calculate_radial_profile(
#     imagepath,
# ):
#     return
