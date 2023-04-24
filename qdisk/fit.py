from os import remove
import casatasks
from .utils import remove_casalogfile, plot_2D_map
remove_casalogfile()
from astropy.coordinates import SkyCoord
from .classes import FitsImage, CasaImage
from .plot import Map
import astropy.units as u
import numpy as np
import pprint
import matplotlib.pyplot as plt


def imfit_wrapper(
    imagename,
    region="",
    mask="", 
    model="",
    residual="",
    estimates="",
    logfile="",
    rms=-1,
    comp_name_list=None,
    print_result=True,
    plot_result=True,
    plot_kw=dict(),
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

    print("Fitting 2D Gaussian to {:s}...".format(imagename))
    result = casatasks.imfit(imagename, region=region, mask=mask, model=model, residual=residual, estimates=estimates, logfile=logfile, rms=rms)
    print("Done!")

    if not result["converged"]:
        print("Fit not converged!. Try again with different parameters.")
    else:
        print("Fit converged.")

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
        c = SkyCoord(ra=ra, dec=dec, frame="icrs")
        r["peak"] = c.ra.to_string(u.hour) + " " + c.dec.to_string(u.degree, alwayssign=True)

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
        incl = np.rad2deg(np.arccos(min / maj)).value % 90
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

    if plot_result:
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9,3), constrained_layout=True)

        # data
        if not imagename.endswith(".fits"):
            casatasks.exportfits(imagename=imagename, fitsimage=imagename+".fits", overwrite=True, dropdeg=True)
            imagename += ".fits"
        ax = axes[0]
        obsmap = Map(imagename, ax=ax, xlim=plot_kw.pop("xlim", None), ylim=plot_kw.pop("ylim", None))
        obsmap.plot_colormap(**plot_kw)
        obsmap.add_beam()
        obsmap.add_colorbar()
        # obsimage.get_directional_coord()
        # plot_2D_map(obsimage.data, ax=ax, X=obsimage.x, Y=obsimage.y, contour=False, beam=obsimage.beam, title="Data", **plot_kw)
        ax.set(xlabel="$\Delta$R.A. [arcsec]", ylabel="$\Delta$Dec. [arcsec]")

        # region
        # fit_region = Regions.parse(region + ' coord=' + header['RADESYS'].lower(), format='crtf')[0]
        # fit_region.to_pixel(WCS(header)).plot(ax=ax, facecolor="none", edgecolor="white", linestyle="dashed")

        # model
        # convert to fits
        casatasks.exportfits(imagename=model, fitsimage=model+".fits", overwrite=True, dropdeg=True)
        ax = axes[1]
        modelmap = Map(model+".fits", ax=ax, xlim=plot_kw.pop("xlim", None), ylim=plot_kw.pop("ylim", None))
        modelmap.plot_colormap(**plot_kw)
        modelmap.add_beam()
        modelmap.add_colorbar()

        # modelimage = CasaImage(model)
        # modelimage.get_directional_coord()
        # plot_2D_map(modelimage.data, ax=ax, X=modelimage.x, Y=modelimage.y, contour=False, beam=obsimage.beam, title="Model", **plot_kw)
        # ax.set(aspect=1./ax.get_data_ratio())

        # # region
        # fit_region = Regions.parse(region + ' coord=' + header['RADESYS'].lower(), format='crtf')[0]
        # fit_region.to_pixel(WCS(header)).plot(ax=ax, facecolor="none", edgecolor="white", linestyle="dashed")
        # # visual clarity
        # ax.tick_params(axis="x", labelbottom=False)  # remove ticklabels for visual clarity
        # ax.tick_params(axis="y", labelleft=False)

        # residual
        casatasks.exportfits(imagename=residual, fitsimage=residual+".fits", overwrite=True, dropdeg=True)
        ax = axes[2]
        modelmap = Map(residual+".fits", ax=ax, xlim=plot_kw.pop("xlim", None), ylim=plot_kw.pop("ylim", None))
        modelmap.plot_colormap(cmap="RdBu_r", vmin=-3*rms, vmax=3*rms)
        modelmap.add_beam()
        modelmap.add_colorbar()
        # residualimage = CasaImage(residual)
        # residualimage.get_directional_coord()
        # plot_kw.update(dict(cmap_kw={
        #     "cmap": "RdBu_r",
        #     "vmin": -3 * rms,
        #     "vmax": 3 * rms,
        # }))
        # plot_2D_map(residualimage.data, ax=ax, X=residualimage.x, Y=residualimage.y, contour=False, beam=obsimage.beam, title="Residual", **plot_kw)
        # ax.set(aspect=1./ax.get_data_ratio())
        # plot

        # plt.subplots_adjust(wspace=0.4)

    return output_result, fig