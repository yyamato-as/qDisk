from astropy.units.quantity import Quantity
from black import mask_cell
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse, AnchoredSizeBar
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes
import numpy as np
import astropy.units as u
import astropy.constants as ac
import pprint
# import casatasks
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
from astropy.wcs import WCS
# from regions import Regions

nologfile = True

c = ac.c.cgs.value
k_B = ac.k_B.cgs.value
h = ac.h.cgs.value
ckms = ac.c.to(u.km/u.s).value

arcsec = np.pi / 180. / 3600. # in rad

# if nologfile:
#     import os 
#     os.system("rm " + os.getcwd() + "/casa-*.log")

### useful functions ###

def is_within(value, range, include_edge=True):
    if include_edge:
        return np.logical_and(value >= range[0], value <= range[1])
    else:
        return np.logical_and(value > range[0], value < range[1])

### unit conversion ###

def get_beam_solid_angle(beam):
    """Calculate the beam solid angle.

    Parameters
    ----------
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec. 

    Returns
    -------
    float
        Beam solid angle in steradian.
    """
    return np.multiply(*beam) * arcsec ** 2 * np.pi / (4 * np.log(2))

def jypb_to_jypsr(I, beam):
    """Convert intensity in Jy / beam to Jy / sr.

    Parameters
    ----------
    I : float or array_like
        Intensity in Jy / beam.
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec. 

    Returns
    -------
    float or array_like
        Intensity in Jy / sr.
    """
    Omega_beam = get_beam_solid_angle(beam)
    return I / Omega_beam

def jypb_to_cgs(I, beam):
    """Convert intensity in Jy / beam to cgs unit. Jy = 1e-23 erg/s/cm2/Hz

    Parameters
    ----------
    I : float or array_like
        Intensity in Jy / beam.
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec. 

    Returns
    -------
    float or array_like
        Intensity in erg s-1 cm-2 Hz-1 sr-1.
    """
    return jypb_to_jypsr(I, beam) * 1e-23

def jypb_to_K_RJ(I, nu, beam):
    """Convert intensity in Jy / beam to birghtness temeprature in Kelvin using RJ approximation.

    Parameters
    ----------
    I : float or array_like
        Intensity in Jy / beam.
    nu : float or array_like
        Observing frequency in Hz.
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec. 

    Returns
    -------
    float or array_like
        Brightness temperature in RJ approximation.
    """
    I = jypb_to_cgs(I, beam)
    return c**2 / (2 * k_B * nu**2) * I

def jypb_to_K(I, nu, beam):
    """Convert intensity in Jy /beam to brightness temperature in Kelvin using full planck function.

    Parameters
    ----------
    I : float or array_like
        Intenisty in Jy /beam.
    nu : flaot or array_like
        Observing frequency in Hz.
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec.

    Returns
    -------
    float or array_like
        Brightness temperature.
    """
    T = np.abs(jypb_to_cgs(I, beam))
    T = h * nu / k_B / np.log(1 + 2 * h * nu**3 / (c**2 * T))

    if isinstance(I, np.ndarray):
        return np.where(T >= 0.0, T, -T)
    elif isinstance(I, float):
        return T if I >= 0.0 else -T

def jypb_to_K_astropy(I, nu, beam):
    """Convert intensity in Jy /beam to brightness temperature in Kelvin using RJ approximation implemented in astropy.

    Parameters
    ----------
    I : float or array_like
        Intenisty in Jy /beam.
    nu : flaot or array_like
        Observing frequency in Hz.
    beam : tuple
        A tuple of beam size, i.e., (bmaj, bmin) in arcsec.

    Returns
    -------
    float or array_like
        Brightness temperature.
    """
    I *= u.Jy / u.beam
    nu *= u.Hz
    Omega_beam = np.multiply(*beam) * u.arcsec ** 2 * np.pi / (4 * np.log(2))
    return I.to(u.K, equivalencies=u.brightness_temperature(nu, beam_area=Omega_beam)).value

### analytical function ###

def Gaussian1d(r, F, sigma):
    return 10 ** F / (np.sqrt(2 * np.pi) * sigma) * np.exp(-r**2 / (2 * sigma**2))

def GaussianRing1d(r, r0, F, sigma):
    return 10 ** F / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(r - r0)**2 / (2 * sigma**2))

def FourthPowerGaussian1d(r, F, sigma):
    return 10 ** F / (np.sqrt(2 * np.pi) * sigma) * np.exp(-r**4 / (2 * sigma**4))

def FourthPowerGaussianRing1d(r, r0, F, sigma):
    return 10 ** F / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(r - r0)**4 / (2 * sigma**4))
    

### fits relevant ###

def fetch_beam_info(header, pa_rotate=False):
    """Fetch the beam information in a FITS file.

    Args:
        header (str): FITS header.

    Returns:
        tuple: beam info in units of arcsec.
    """
    # header = fits.getheader(fits_filename)
    bmaj = header['BMAJ']*3600
    bmin = header['BMIN']*3600
    bpa = header['BPA']
    if pa_rotate: # for plot_2D_map; checked the pa with casaviewer by eye
        bpa = 90 - bpa
    return bmaj, bmin, bpa

### plot functions ###


def add_beam(ax=None, beam=(0.5, 0.5, 0.0), **kwargs):
    """Add a beam ellipse onto an axis.

    Args:
        ax (axes, optional): The axis onto which a beam ellipse is draw. Defaults to None.
        beam (tuple[float], optional): Beam major axis, minor axis, and position angle (degrees). Need to be in the same unit as ax. position angle is measured from north (positive y) to east (positive x). Defaults to (0.5, 0.5, 0.0).
    """

    if ax is None:
        fig, ax = plt.subplots()
    bmaj, bmin, bpa = beam

    width = bmaj
    height = bmin
    angle = 90 - bpa # to make measured from east 
    beam = AnchoredEllipse(
        ax.transData, width=width, height=height, angle=angle, loc="lower left", pad=0.5, borderpad=0.5, frameon=False,
    )
    beam.ellipse.set(color=kwargs.get('color', 'white'), fill=kwargs.get('fill', False), hatch=kwargs.get('hatch', "//////"))
    ax.add_artist(beam)

    return


def add_scalebar(ax=None, scale=50, text=None, **kwargs):
    """Add a scale bar onto an axis.

    Args:
        ax (axes, optional): The axis onto which a scale bar is draw. Defaults to None.
        scale (int, optional): The scale for which a scale bar to be added. Need to be in the same sacling as ax. Defaults to 50.
        text (str, optional): Annotation text. Defaults to None.
    """

    if ax is None:
        fig, ax = plt.subplots()

    size = scale
    label = text
    scalebar = AnchoredSizeBar(
        ax.transData,
        size=size,
        label=label,
        loc=kwargs.get('loc', "lower right"),
        pad=kwargs.get('pad', 0.1),
        borderpad=kwargs.get('borderpad', 0.5),
        sep=kwargs.get('sep', 3),
        frameon=kwargs.get('frameon', False),
        color=kwargs.get('color', 'white'),
        fontproperties=fm.FontProperties(size=9),
    )
    ax.add_artist(scalebar)
    return


def set_colorbar_extend(image, data):
    if image.norm.vmax >= np.nanmax(data) and image.norm.vmin <= np.nanmin(data):
        return "neither"
    elif image.norm.vmax < np.nanmax(data) and image.norm.vmin <= np.nanmin(data):
        return "max"
    elif image.norm.vmax >= np.nanmax(data) and image.norm.vmin > np.nanmin(data):
        return "min"
    else:
        return "both"


def plot_2D_map(
    data,
    X=None,
    Y=None,
    ax=None,
    cmap=True,
    cmap_method="contourf",
    contour=True,
    colorbar=True,
    title=None,
    beam=None,
    scale=None,
    cmap_kw={},
    contour_kw={},
    cbar_kw={},
    beam_kw={},
    sbar_kw={}
):
    
    if ax is None:
        fig, ax = plt.subplots()

    # colormap
    if cmap:
        plot = getattr(ax, cmap_method)
        image = plot(X, Y, data, rasterized=True, **cmap_kw)
    if contour:
        ax.contour(X, Y, data, **contour_kw)
    ax.invert_xaxis()

    # add beam and scalebar and title
    if beam is not None:
        add_beam(ax, beam, **beam_kw)
    if scale is not None:
        add_scalebar(ax, scale=scale[0], text=scale[1], **sbar_kw)
    ax.set_title(title)

    # colorbar
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position="right", size="5%", axes_class=maxes.Axes, pad=0.1)
        extend = set_colorbar_extend(image, data)
        fig = ax.get_figure()
        fig.colorbar(image, cax=cax, extend=extend, **cbar_kw)

    # ax.set_aspect(1./ax.get_data_ratio())

    return 

def plot_1D_profile(x, y, yerr=None, ax=None, color="black", style="line", label=None):
    if ax is None:
        fig, ax = plt.subplots()

    if yerr is None:
        if style == "line":
            ax.plot(x, y, color=color, label=label)
        elif style == "step":
            ax.plot(x, y, color=color, drawstyle="steps-mid", label=label)
        elif style == "point":
            ax.plot(x, y, fmt="o", color=color, label=label)
    else:
        if style == "line":
            ax.plot(x, y, color=color, label=label)
            ax.fill_between(x, y-yerr, y+yerr, alpha=0.25, edgecolor=None, color=color)
        elif style == "step":
            ax.errorbar(x, y, yerr=yerr, drawstyle="steps-mid", color=color)
        elif style == "point":
            ax.errorbar(x, y, yerr=yerr, fmt="o", color=color)

    ax.set(xlim=(x.min(), x.max()))

    return fig, ax




#def plot_channel_maps(header, data)


### 2D Gaussian fit on the image plane; wrapper for the CASA imfit ###


def imfit_wrapper(
    imagepath,
    region="",
    model="",
    residual="",
    estimates="",
    rms=-1,
    comp_name_list=None,
    print_result=True,
    plot=True,
    plot_region_slices=(),
    plot_kw={},
):
    """A wrapper for CASA imfit task to fit one or more Gaussian component(s) to an image.

    Args:
        imagepath (str): Path to the FITS file.
        region (str, optional): Fit region with the CASA Region format. Defaults to use the full image.
        model (str, optional): Path to output model image. Defaults not to output any model image file.
        residual (str, optional): Path to output residual image. Defaults not to output any residual image file.
        estimates (str, optional): Path to input initial estimates file with the CASA estimates format. Defaults not to use nay initial guesses.
        rms (any, optional): The image rms to be used for the error calculation. Defaults (or any negative values) to use the rms of residual image.
        comp_name_list (list, optional): Component name list for output. Defaults to None.
        print_result (bool, optional): If print the fit result or not. Defaults to True.
        plot (bool, optional): If plot the data, model, and residual. Defaults to True.
        plot_region_slices (tuple, optional): Relevant Only when plot = True. Define the plot region by a pair of slices. Defaults to plot the full image.
        plot_kw (dict, optional): kwargs passed to .plot_2D_map. Defaults to {}.

    Returns:
        dict: A dictionary contains the fit result, i.e., fitted parameters.
    """

    print("Start fitting 2D Gaussian to {:s}...".format(imagepath))
    result = casatasks.imfit(imagepath, region=region, model=model, residual=residual, estimates=estimates, rms=rms)
    print("Done!")

    if not result["converged"]:
        print("Fit not converged. Try again with different parameters.")
    else:
        print("Fit converged!")

    if comp_name_list is None:
        comp_name_list = ["component{:d}".format(i) for i in range(result["deconvolved"]["nelements"])]

    # rearrange the result dictionary for easy use
    output_result = {}
    for i, comp in enumerate(comp_name_list):
        output_result[comp] = {}
        r = output_result[comp]

        # point source or Gaussian
        r["ispoint"] = result["results"]["component{:d}".format(i)]["ispoint"]

        # peak coordinate
        ra = result["results"]["component{:d}".format(i)]["shape"]["direction"]["m0"]["value"] * u.Unit(
            result["results"]["component{:d}".format(i)]["shape"]["direction"]["m0"]["unit"]
        )
        dec = result["results"]["component{:d}".format(i)]["shape"]["direction"]["m1"]["value"] * u.Unit(
            result["results"]["component{:d}".format(i)]["shape"]["direction"]["m1"]["unit"]
        )
        frame = result["results"]["component{:d}".format(i)]["shape"]["direction"]["refer"].lower()
        c = SkyCoord(ra=ra, dec=dec, frame=frame)
        r["peak"] = c

        # size
        maj = result["results"]["component{:d}".format(i)]["shape"]["majoraxis"]["value"] * u.Unit(
            result["results"]["component{:d}".format(i)]["shape"]["majoraxis"]["unit"]
        )
        min = result["results"]["component{:d}".format(i)]["shape"]["minoraxis"]["value"] * u.Unit(
            result["results"]["component{:d}".format(i)]["shape"]["minoraxis"]["unit"]
        )
        pa = result["results"]["component{:d}".format(i)]["shape"]["positionangle"]["value"] * u.Unit(
            result["results"]["component{:d}".format(i)]["shape"]["positionangle"]["unit"]
        )
        r["size"] = {}
        r["size"]["maj"] = maj
        r["size"]["min"] = min
        r["size"]["pa"] = pa

        # calculate inclination
        incl = np.rad2deg(np.arccos(min / maj)).value % 360
        r["inclination"] = incl * u.deg

        # flux
        r["flux"] = result["results"]["component{:d}".format(i)]["flux"]["value"][0] * u.Unit(
            result["results"]["component{:d}".format(i)]["flux"]["unit"]
        )
        r["flux_error"] = result["results"]["component{:d}".format(i)]["flux"]["error"][0] * u.Unit(
            result["results"]["component{:d}".format(i)]["flux"]["unit"]
        )

        # peak intensity
        r["peak_intensity"] = result["results"]["component{:d}".format(i)]["peak"]["value"] * u.Unit(
            result["results"]["component{:d}".format(i)]["peak"]["unit"]
        )
        r["peak_intensity_error"] = result["results"]["component{:d}".format(i)]["peak"]["error"] * u.Unit(
            result["results"]["component{:d}".format(i)]["peak"]["unit"]
        )

    if print_result:
        pprint.pprint(output_result)

    if plot:
        fig = plt.figure(figsize=(12, 4))

        # data
        header = fits.getheader(imagepath)
        data = fits.getdata(imagepath)
        beam = (header['BMAJ']/np.abs(header['CDELT1']), header['BMIN']/np.abs(header['CDELT2']), header['BPA'] + 90)
        ax = fig.add_subplot(131, projection=WCS(header))
        plot_2D_map(data[plot_region_slices], ax=ax, contour=False, beam=beam, title="Data", **plot_kw)

        # region
        fit_region = Regions.parse(region + ' coord=' + header['RADESYS'].lower(), format='crtf')[0]
        fit_region.to_pixel(WCS(header)).plot(ax=ax, facecolor="none", edgecolor="white", linestyle="dashed")

        # model
        # export to FITS and read it
        modfits = model + ".fits"
        casatasks.exportfits(model, fitsimage=modfits, overwrite=True, dropdeg=True)
        header = fits.getheader(modfits)
        data = fits.getdata(modfits)

        # plot
        ax = fig.add_subplot(132, projection=WCS(header))
        plot_2D_map(data[plot_region_slices], ax=ax, contour=False, title="Model", **plot_kw)

        # region
        fit_region = Regions.parse(region + ' coord=' + header['RADESYS'].lower(), format='crtf')[0]
        fit_region.to_pixel(WCS(header)).plot(ax=ax, facecolor="none", edgecolor="white", linestyle="dashed")
        # visual clarity
        ax.tick_params(axis="x", labelbottom=False)  # remove ticklabels for visual clarity
        ax.tick_params(axis="y", labelleft=False)

        # residual
        # export to FITS and read it
        resfits = residual + ".fits"
        casatasks.exportfits(residual, fitsimage=resfits, overwrite=True, dropdeg=True)
        header = fits.getheader(resfits)
        data = fits.getdata(resfits)

        # plot
        ax = fig.add_subplot(133, projection=WCS(header))
        plot_kw["imshow_kw"] = {
            "cmap": "RdBu_r",
            "vmin": -3 * rms,
            "vmax": 3 * rms,
        }  # change to diverging cmap and rms limited color range
        plot_2D_map(data[plot_region_slices], ax=ax, contour=False, title="Residual", **plot_kw)

        # region
        fit_region = Regions.parse(region + ' coord=' + header['RADESYS'].lower(), format='crtf')[0]
        fit_region.to_pixel(WCS(header)).plot(ax=ax, facecolor="none", edgecolor="black", linestyle="dashed")
        # visual clarity
        ax.tick_params(axis="x", labelbottom=False)  # remove ticklabels for visual clarity
        ax.tick_params(axis="y", labelleft=False)

        plt.subplots_adjust(wspace=0.4)

    return output_result


### relevant to image analysis ###

def get_radec_coord(header, center_coord=None):
    """Generate a (RA\cos(Dec), Dec) coordinate (1D each) in arcsec. Assume the unit for coordinates in the header is deg.

    Args:
        header (dict): FITS header.
        center_coord (tuple or astropy.coordinates.SkyCoord object, optinal): Two component tuple of (RA, Dec) in arcsec or the SkyCoord object for the center coordinate. Defaults to (0.0, 0.0)

    Returns:
        tuple: Coordinates
    """

    if center_coord is None:
        offset_x, offset_y = 0, 0
    else:
        if isinstance(center_coord, tuple):
            center_x, center_y = center_coord
        elif isinstance(center_coord, SkyCoord):
            center_x = center_coord.ra.arcsec 
            center_y = center_coord.dec.arcsec
        offset_x = center_x - header['CRVAL1'] * 3600 # offset along x from phsecenter in arcsec
        offset_y = center_y - header['CRVAL2'] * 3600 # offset along y from phsecenter in arcsec

    dx = header['CDELT1'] * 3600 # x increment 
    dy = header['CDELT2'] * 3600 # y increment 

    npix = header['NAXIS1']
    assert header['NAXIS1'] == header['NAXIS2'] # assert image is square

    x = dx * (np.arange(npix) - (header['CRPIX1'] - 1)) - offset_x
    y = dy * (np.arange(npix) - (header['CRPIX2'] - 1)) - offset_y

    return x, y

def get_spectral_coord(header, which='both'):
    if header['NAXIS'] < 3:
        raise KeyError("Spectral axis not found.")
    
    nchan = header['NAXIS3']
    delta = header['CDELT3'] # assume in Hz
    nu = delta * (np.arange(nchan) - (header['CRPIX3'] - 1)) + header['CRVAL3']

    if which == 'freq':
        return nu

    assert header['VELREF'] == 257
    v = ckms * (1 - nu / header['RESTFRQ'])

    if which == 'vel':
        return v
    
    if which == 'both':
        return nu, v
    


def get_projected_coord(header, PA=0., incl=45., center_coord=None, which='both'): 
    x, y = get_radec_coord(header, center_coord=center_coord)

    # meshgrid to be in 2D
    xx, yy = np.meshgrid(x, y)

    # project to the disk plane; assume geometrically thin disk
    incl = np.radians(incl)
    PA = np.radians(PA)

    x_proj = (xx * np.sin(PA) + yy * np.cos(PA)) 
    y_proj = (- xx * np.cos(PA) + yy * np.sin(PA)) / np.cos(incl) # follow the formulation in Yen et al. 2016

    if which == 'cartesian':
        return x_proj, y_proj

    # polar coordinate
    r = np.sqrt(x_proj**2 + y_proj**2) # in arcsec
    theta = np.degrees(np.arctan2(y_proj, x_proj)) # in degree, [-180, 180]

    if which == 'polar':
        return r, theta
    
    if which == 'both':
        return (x_proj, y_proj), (r, theta)

# def plot_projected_coord(x, y, projected_coords, ax=None, data=None, plot_kw={}):
#     if ax is None:
#         fig, ax = plt.subplots()

#     a, b = projected_coords

#     if data is not None:
#         plot_2D_map(data, X=x, Y=y, ax=ax, contour=False, **plot_kw)
    
#     ax.contour(x, y, a, lavels=20, )


# TODO: there is a better way to calculate the profile; not to use for loop, instead use binned_statistics in scipy
def calc_radial_profile(header, data, PA=0., incl=45., center_coord=(0., 0.), rbins=None, rmin=0.0, rmax=None, include_theta=180.):

    r, theta = get_projected_coord(header, PA=PA, incl=incl, center_coord=center_coord, which='polar')

    if rbins is None:
        rbin_width = header['BMAJ'] * 3600 * 0.25
        if rmax is None:
            rmax = np.nanmax(r)
        rbins = np.arange(rmin, rmax, rbin_width)
    
    else:
        rbin_width = np.diff(rbins)[-1]

    rad_prof = np.zeros(len(rbins))
    rad_prof_scatter = np.zeros_like(rad_prof)

    theta_exclude = ((theta > -180. + 0.5*include_theta) & (theta < -0.5*include_theta)) | ((theta > 0.5*include_theta) & (theta < 180. - 0.5*include_theta))

    for i in range(len(rbins)):
        r_include = (r >= rbins[i] - 0.5*rbin_width) & (r <= rbins[i] + 0.5*rbin_width)
        az_sample = data[r_include & ~theta_exclude]

        rad_prof[i] = np.average(az_sample)
        rad_prof_scatter[i] = np.std(az_sample)
    
    return rbins, rad_prof, rad_prof_scatter
    

def sigma_to_FWHM(sigma):
    return sigma * np.sqrt(8 * np.log(2))


def FWHM_to_sigma(FWHM):
    return FWHM / np.sqrt(8 * np.log(2))

#def slice_image(data, xlim=(), ylim=(), zlim=()):

# def calc_rms(data, axis=None):

    

def calc_rms(data, axis=None, ignore_zero=True, sigma_th=1.0):
    if ignore_zero:
        mask = np.logical_and(~np.isnan(data), data != 0.0)
        ndata = np.count_nonzero(mask, axis=axis) # neither 0 nor nan
        mean = np.sum(data, axis=axis, where=mask) / ndata
        delta = data - mean
        np.copyto(delta, 0., where=~mask)
        std = np.sqrt(np.sum(delta**2, axis=axis, where=mask) / ndata)
    else:
        mean = np.nanmean(data, axis=axis)
        std = np.nanstd(data, axis=axis)
    if np.abs(mean) >= sigma_th * std:
        print("The mean of data is deviated by >{:.1f}sigma from zero. Will use stddev instead.".format(sigma_th))
        return std
    else:
        if ignore_zero:
            return np.sqrt(np.sum(data**2, axis=axis, where=mask) / ndata)
        else:
            return np.sqrt(np.nansum(data**2, axis=axis) / np.count_nonzero(~np.isnan(data), axis=axis))


def get_mask(r, theta, rmin=0.0, rmax=100.0, thetamin=-180., thetamax=180., combine="and"):
    r_mask = np.where(((r >= rmin) & (r <= rmax)), 1.0, 0.0)
    theta_mask = np.where(((theta >= thetamin) & (theta <= thetamax)), 1.0, 0.0)
    combine = np.logical_and if combine == "and" else np.logical_or
    mask = combine(r_mask, theta_mask)
    return mask
    
    








# """
# from astropy.units.quantity import Quantity
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse, AnchoredSizeBar
# import matplotlib.font_manager as fm
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.axes as maxes
# import numpy as np
# import astropy.units as u
# import astropy.constants as ac
# import pprint
# import casatasks
# from astropy.coordinates import SkyCoord
# import astropy.io.fits as fits
# from astropy.wcs import WCS
# from regions import Regions

# ckms = ac.c.to(u.km/u.s).value

# ### fits relevant ###

# def fetch_beam_info(fits_filename):
#     """Fetch the beam information in a FITS file.

#     Args:
#         fits_filename (str): FITS file path.

#     Returns:
#         tuple: beam info in units of arcsec.
#     """
#     header = fits.getheader(fits_filename)
#     bmaj = header['BMAJ']*3600
#     bmin = header['BMIN']*3600
#     bpa = header['BPA']
#     return bmaj, bmin, bpa

# ### plot functions ###


# def add_beam(ax=None, beam=(0.5, 0.5, 0.0), **kwargs):
#     """Add a beam ellipse onto an axis.

#     Args:
#         ax (axes, optional): The axis onto which a beam ellipse is draw. Defaults to None.
#         beam (tuple[float], optional): Beam major axis, minor axis, and position angle (degrees). Need to be in the same unit as ax. position angle is measured from positive x axis to positive y. Defaults to (0.5, 0.5, 0.0).
#     """

#     if ax is None:
#         fig, ax = plt.subplots()
#     bmaj, bmin, bpa = beam

#     width = bmaj
#     height = bmin
#     angle = bpa 
#     beam = AnchoredEllipse(
#         ax.transData, width=width, height=height, angle=angle, loc="lower left", pad=0.5, borderpad=0.5, frameon=False,
#     )
#     beam.ellipse.set(color=kwargs.get('color', 'white'), fill=kwargs.get('fill', False), hatch=kwargs.get('hatch', "//////"))
#     ax.add_artist(beam)

#     return


# def add_scalebar(ax=None, scale=50, text=None, **kwargs):
#     """Add a scale bar onto an axis.

#     Args:
#         ax (axes, optional): The axis onto which a scale bar is draw. Defaults to None.
#         scale (int, optional): The scale for which a scale bar to be added. Need to be in the same sacling as ax. Defaults to 50.
#         text (str, optional): Annotation text. Defaults to None.
#     """

#     if ax is None:
#         fig, ax = plt.subplots()

#     size = scale
#     label = text
#     scalebar = AnchoredSizeBar(
#         ax.transData,
#         size=size,
#         label=label,
#         loc=kwargs.get('loc', "lower right"),
#         pad=kwargs.get('pad', 0.1),
#         borderpad=kwargs.get('borderpad', 0.5),
#         sep=kwargs.get('sep', 3),
#         frameon=kwargs.get('frameon', False),
#         color=kwargs.get('color', 'white'),
#         fontproperties=fm.FontProperties(size=9),
#     )
#     ax.add_artist(scalebar)
#     return


# def set_colorbar_extend(image, data):
#     if image.norm.vmax >= np.nanmax(data) and image.norm.vmin <= np.nanmin(data):
#         return "neither"
#     elif image.norm.vmax < np.nanmax(data) and image.norm.vmin <= np.nanmin(data):
#         return "max"
#     elif image.norm.vmax >= np.nanmax(data) and image.norm.vmin > np.nanmin(data):
#         return "min"
#     else:
#         return "both"


# def plot_2D_map(
#     data,
#     X=None,
#     Y=None,
#     ax=None,
#     contour=True,
#     colorbar=True,
#     title=None,
#     beam=None,
#     scale=None,
#     imshow_kw={},
#     pcolorfast_kw={},
#     contour_kw={},
#     cbar_kw={},
#     beam_kw={},
#     sbar_kw={}
# ):
#     """Plot an acceptable 2D emission map.

#     Args:
#         data (array-like): data which to draw.
#         X (array-like, optional): X-coordinate for pcolormesh. 2D (meshgrid) or 1D. If None, use imshow which is faster instead pcolormesh. Defaults to None.
#         Y (array-like, optional): Y-coordinate for pcolormesh. 2D (meshgrid) or 1D. If None, use imshow which is faster instead pcolormesh. Defaults to None.
#         ax (axes, optional): The axis onto which the map is draw. Defaults to None.
#         contour (bool, optional): If contour is draw or not. Defaults to True.
#         colorbar (bool, optional): If colrobar is added or not. Defaults to True.
#         title (str, optional): Title for the axis. Defaults to None.
#         beam (tuple or None, optional): Beam ellipse info to be added onto the axis. Beam major axis, minor axis, and position angle (degrees). Need to be in the same unit as ax. position angle is measured from positive x axis to positive y. Defaults to None (beam ellipse not added).
#         scale (tuple or None, optional): Scale bar info. First of tuple should be the float representing the scale in the same unit as ax, and the second to be the annotation text. Defaults to None (scalebar not added).  
#         imshow_kw (dict, optional): kwargs passed to imshow. Defaults to {}.
#         pcolorfast_kw (dict, optional): kwargs passed to pcolorfast. Defaults to {}.
#         contour_kw (dict, optional): kwargs passed to contour. Defaults to {}.
#         cbar_kw (dict, optional): kwargs passed to colorbar. Defaults to {}.
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     if X is None or Y is None:
#         image = ax.imshow(data, origin="lower", **imshow_kw)
#         if contour:
#             ax.contour(data, **contour_kw)
#     else:
#         image = ax.pcolorfast(X, Y, data, rasterized=True, **pcolorfast_kw)
#         ax.invert_xaxis()
#         if contour:
#             ax.contour(X, Y, data, **contour_kw)

#     # add beam and scalebar and title
#     if beam is not None:
#         add_beam(ax, beam, **beam_kw)
#     if scale is not None:
#         add_scalebar(ax, scale=scale[0], text=scale[1], **sbar_kw)
#     ax.set_title(title)

#     # colorbar
#     if colorbar:
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes(position="right", size="5%", axes_class=maxes.Axes, pad=0.25)
#         extend = set_colorbar_extend(image, data)
#         fig = ax.get_figure()
#         fig.colorbar(image, cax=cax, extend=extend, **cbar_kw)

#     ax.set_aspect(1./ax.get_data_ratio())

#     return


# #def plot_channel_maps(header, data)


# ### 2D Gaussian fit on the image plane; wrapper for the CASA imfit ###


# def imfit_wrapper(
#     imagepath,
#     region="",
#     model="",
#     residual="",
#     estimates="",
#     rms=-1,
#     comp_name_list=None,
#     print_result=True,
#     plot=True,
#     plot_region_slices=(),
#     plot_kw={},
# ):
#     """A wrapper for CASA imfit task to fit one or more Gaussian component(s) to an image.

#     Args:
#         imagepath (str): Path to the FITS file.
#         region (str, optional): Fit region with the CASA Region format. Defaults to use the full image.
#         model (str, optional): Path to output model image. Defaults not to output any model image file.
#         residual (str, optional): Path to output residual image. Defaults not to output any residual image file.
#         estimates (str, optional): Path to input initial estimates file with the CASA estimates format. Defaults not to use nay initial guesses.
#         rms (any, optional): The image rms to be used for the error calculation. Defaults (or any negative values) to use the rms of residual image.
#         comp_name_list (list, optional): Component name list for output. Defaults to None.
#         print_result (bool, optional): If print the fit result or not. Defaults to True.
#         plot (bool, optional): If plot the data, model, and residual. Defaults to True.
#         plot_region_slices (tuple, optional): Relevant Only when plot = True. Define the plot region by a pair of slices. Defaults to plot the full image.
#         plot_kw (dict, optional): kwargs passed to .plot_2D_map. Defaults to {}.

#     Returns:
#         dict: A dictionary contains the fit result, i.e., fitted parameters.
#     """

#     print("Start fitting 2D Gaussian to {:s}...".format(imagepath))
#     result = casatasks.imfit(imagepath, region=region, model=model, residual=residual, estimates=estimates, rms=rms)
#     print("Done!")

#     if not result["converged"]:
#         print("Fit not converged. Try again with different parameters.")
#     else:
#         print("Fit converged!")

#     if comp_name_list is None:
#         comp_name_list = ["component{:d}".format(i) for i in range(result["deconvolved"]["nelements"])]

#     # rearrange the result dictionary for easy use
#     output_result = {}
#     for i, comp in enumerate(comp_name_list):
#         output_result[comp] = {}
#         r = output_result[comp]

#         # point source or Gaussian
#         r["ispoint"] = result["results"]["component{:d}".format(i)]["ispoint"]

#         # peak coordinate
#         ra = result["results"]["component{:d}".format(i)]["shape"]["direction"]["m0"]["value"] * u.Unit(
#             result["results"]["component{:d}".format(i)]["shape"]["direction"]["m0"]["unit"]
#         )
#         dec = result["results"]["component{:d}".format(i)]["shape"]["direction"]["m1"]["value"] * u.Unit(
#             result["results"]["component{:d}".format(i)]["shape"]["direction"]["m1"]["unit"]
#         )
#         frame = result["results"]["component{:d}".format(i)]["shape"]["direction"]["refer"].lower()
#         c = SkyCoord(ra=ra, dec=dec, frame=frame)
#         r["peak"] = c

#         # size
#         maj = result["results"]["component{:d}".format(i)]["shape"]["majoraxis"]["value"] * u.Unit(
#             result["results"]["component{:d}".format(i)]["shape"]["majoraxis"]["unit"]
#         )
#         min = result["results"]["component{:d}".format(i)]["shape"]["minoraxis"]["value"] * u.Unit(
#             result["results"]["component{:d}".format(i)]["shape"]["minoraxis"]["unit"]
#         )
#         pa = result["results"]["component{:d}".format(i)]["shape"]["positionangle"]["value"] * u.Unit(
#             result["results"]["component{:d}".format(i)]["shape"]["positionangle"]["unit"]
#         )
#         r["size"] = {}
#         r["size"]["maj"] = maj
#         r["size"]["min"] = min
#         r["size"]["pa"] = pa

#         # calculate inclination
#         incl = np.rad2deg(np.arccos(min / maj)).value % 360
#         r["inclination"] = incl * u.deg

#         # flux
#         r["flux"] = result["results"]["component{:d}".format(i)]["flux"]["value"][0] * u.Unit(
#             result["results"]["component{:d}".format(i)]["flux"]["unit"]
#         )
#         r["flux_error"] = result["results"]["component{:d}".format(i)]["flux"]["error"][0] * u.Unit(
#             result["results"]["component{:d}".format(i)]["flux"]["unit"]
#         )

#         # peak intensity
#         r["peak_intensity"] = result["results"]["component{:d}".format(i)]["peak"]["value"] * u.Unit(
#             result["results"]["component{:d}".format(i)]["peak"]["unit"]
#         )
#         r["peak_intensity_error"] = result["results"]["component{:d}".format(i)]["peak"]["error"] * u.Unit(
#             result["results"]["component{:d}".format(i)]["peak"]["unit"]
#         )

#     if print_result:
#         pprint.pprint(output_result)

#     if plot:
#         fig = plt.figure(figsize=(12, 4))

#         # data
#         header = fits.getheader(imagepath)
#         data = fits.getdata(imagepath)
#         beam = (header['BMAJ']/np.abs(header['CDELT1']), header['BMIN']/np.abs(header['CDELT2']), header['BPA'] + 90)
#         ax = fig.add_subplot(131, projection=WCS(header))
#         plot_2D_map(data[plot_region_slices], ax=ax, contour=False, beam=beam, title="Data", **plot_kw)

#         # region
#         fit_region = Regions.parse(region + ' coord=' + header['RADESYS'].lower(), format='crtf')[0]
#         fit_region.to_pixel(WCS(header)).plot(ax=ax, facecolor="none", edgecolor="white", linestyle="dashed")

#         # model
#         # export to FITS and read it
#         modfits = model + ".fits"
#         casatasks.exportfits(model, fitsimage=modfits, overwrite=True, dropdeg=True)
#         header = fits.getheader(modfits)
#         data = fits.getdata(modfits)

#         # plot
#         ax = fig.add_subplot(132, projection=WCS(header))
#         plot_2D_map(data[plot_region_slices], ax=ax, contour=False, title="Model", **plot_kw)

#         # region
#         fit_region = Regions.parse(region + ' coord=' + header['RADESYS'].lower(), format='crtf')[0]
#         fit_region.to_pixel(WCS(header)).plot(ax=ax, facecolor="none", edgecolor="white", linestyle="dashed")
#         # visual clarity
#         ax.tick_params(axis="x", labelbottom=False)  # remove ticklabels for visual clarity
#         ax.tick_params(axis="y", labelleft=False)

#         # residual
#         # export to FITS and read it
#         resfits = residual + ".fits"
#         casatasks.exportfits(residual, fitsimage=resfits, overwrite=True, dropdeg=True)
#         header = fits.getheader(resfits)
#         data = fits.getdata(resfits)

#         # plot
#         ax = fig.add_subplot(133, projection=WCS(header))
#         plot_kw["imshow_kw"] = {
#             "cmap": "RdBu_r",
#             "vmin": -3 * rms,
#             "vmax": 3 * rms,
#         }  # change to diverging cmap and rms limited color range
#         plot_2D_map(data[plot_region_slices], ax=ax, contour=False, title="Residual", **plot_kw)

#         # region
#         fit_region = Regions.parse(region + ' coord=' + header['RADESYS'].lower(), format='crtf')[0]
#         fit_region.to_pixel(WCS(header)).plot(ax=ax, facecolor="none", edgecolor="black", linestyle="dashed")
#         # visual clarity
#         ax.tick_params(axis="x", labelbottom=False)  # remove ticklabels for visual clarity
#         ax.tick_params(axis="y", labelleft=False)

#         plt.subplots_adjust(wspace=0.4)

#     return output_result


# ### relevant to image analysis ###

# def get_radec_coord(header, center_coord=(0.,0.)):
#     """Generate a (RA\cos(Dec), Dec) coordinate (1D each) in arcsec. Assume the unit for coordinates in the header is deg.

#     Args:
#         header (dict): FITS header.
#         center_coord (tuple or astropy.coordinates.SkyCoord object, optinal): Two component tuple of (RA, Dec) in arcsec or the SkyCoord object for the center coordinate. Defaults to (0.0, 0.0)

#     Returns:
#         tuple: Coordinates
#     """

#     if isinstance(center_coord, tuple):
#         center_x, center_y = center_coord
#     elif isinstance(center_coord, SkyCoord):
#         center_x = center_coord.ra.arcsec 
#         center_y = center_coord.dec.arcsec

#     offset_x = center_x - header['CRVAL1'] * 3600 # offset along x from phsecenter in arcsec
#     offset_y = center_y - header['CRVAL2'] * 3600 # offset along y from phsecenter in arcsec

#     dx = header['CDELT1'] * 3600 # x increment 
#     dy = header['CDELT2'] * 3600 # y increment 

#     npix = header['NAXIS1']
#     assert header['NAXIS1'] == header['NAXIS2'] # assert image is square

#     x = dx * (np.arange(npix) - (header['CRPIX1'] - 1)) - offset_x
#     y = dy * (np.arange(npix) - (header['CRPIX2'] - 1)) - offset_y

#     return x, y

# def get_spectral_coord(header, which='both'):
#     if header['NAXIS'] < 3:
#         raise KeyError("Spectral axis not found.")
    
#     nchan = header['NAXIS3']
#     delta = header['CDELT3'] # assume in Hz
#     nu = delta * (np.arange(nchan) - (header['CRPIX3'] - 1)) + header['CRVAL3']

#     if which == 'freq':
#         return nu

#     assert header['VELREF'] == 257
#     v = ckms * (1 - nu / header['RESTFRQ'])

#     if which == 'vel':
#         return v
    
#     if which == 'both':
#         return nu, v
    


# def get_projected_coord(header, PA=0., incl=45., center_coord=(0., 0.), which='both'): 
#     x, y = get_radec_coord(header, center_coord=center_coord)

#     # meshgrid to be in 2D
#     xx, yy = np.meshgrid(x, y)

#     # project to the disk plane; assume geometrically thin disk
#     incl = np.radians(incl)
#     PA = np.radians(PA)

#     x_proj = (xx * np.sin(PA) + yy * np.cos(PA)) 
#     y_proj = (- xx * np.cos(PA) + yy * np.sin(PA)) / np.cos(incl) # follow the formulation in Yen et al. 2016

#     if which == 'cartesian':
#         return x_proj, y_proj

#     # polar coordinate
#     r = np.sqrt(x_proj**2 + y_proj**2) # in arcsec
#     theta = np.degrees(np.arctan2(y_proj, x_proj)) # in degree, [-180, 180]

#     if which == 'polar':
#         return r, theta
    
#     if which == 'both':
#         return (x_proj, y_proj), (r, theta)

# # def plot_projected_coord(x, y, projected_coords, ax=None, data=None, plot_kw={}):
# #     if ax is None:
# #         fig, ax = plt.subplots()

# #     a, b = projected_coords

# #     if data is not None:
# #         plot_2D_map(data, X=x, Y=y, ax=ax, contour=False, **plot_kw)
    
# #     ax.contour(x, y, a, lavels=20, )


# def calc_radial_profile(header, data, PA=0., incl=45., center_coord=(0., 0.), rbins=None, rmin=0.0, rmax=None, include_theta=180.):

#     r, theta = get_projected_coord(header, PA=PA, incl=incl, center_coord=center_coord, which='polar')

#     if rbins is None:
#         rbin_width = header['BMAJ'] * 3600 * 0.25
#         if rmax is None:
#             rmax = np.nanmax(r)
#         rbins = np.arange(rmin, rmax, rbin_width)
    
#     else:
#         rbin_width = np.diff(rbins)[-1]

#     rad_prof = np.zeros(len(rbins))
#     rad_prof_scatter = np.zeros_like(rad_prof)

#     theta_exclude = ((theta > -180. + 0.5*include_theta) & (theta < -0.5*include_theta)) | ((theta > 0.5*include_theta) & (theta < 180. - 0.5*include_theta))

#     for i in range(len(rbins)):
#         r_include = (r >= rbins[i] - 0.5*rbin_width) & (r <= rbins[i] + 0.5*rbin_width)
#         az_sample = data[r_include & ~theta_exclude]

#         rad_prof[i] = np.average(az_sample)
#         rad_prof_scatter[i] = np.std(az_sample)
    
#     return rbins, rad_prof, rad_prof_scatter
    

# def sigma_to_FWHM(sigma):
#     return sigma * np.sqrt(8 * np.log(2))


# def FWHM_to_sigma(FWHM):
#     return FWHM / np.sqrt(8 * np.log(2))

# #def slice_image(data, xlim=(), ylim=(), zlim=()):
    

# def calc_rms(data, axis=None, ignore_zero=True, sigma_th=1.0):
#     if ignore_zero:
#         mask = np.logical_and(~np.isnan(data), data != 0.0)
#         ndata = np.count_nonzero(mask, axis=axis) # neither 0 nor nan
#         mean = np.sum(data, axis=axis, where=mask) / ndata
#         delta = data - mean
#         np.copyto(delta, 0., where=~mask)
#         std = np.sqrt(np.sum(delta**2, axis=axis, where=mask) / ndata)
#     else:
#         mean = np.nanmean(data, axis=axis)
#         std = np.nanstd(data, axis=axis)
#     if np.abs(mean) >= sigma_th * std:
#         print("The mean of data is deviated by >{:.1f}sigma from zero. Will use stddev instead.".format(sigma_th))
#         return std
#     else:
#         if ignore_zero:
#             return np.sqrt(np.sum(data**2, axis=axis, where=mask) / ndata)
#         else:
#             return np.sqrt(np.nansum(data**2, axis=axis) / np.count_nonzero(~np.isnan(data), axis=axis))

