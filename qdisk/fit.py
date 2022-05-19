import casatasks
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pprint

def imfit_wrapper(
    imagename,
    region="",
    mask="", 
    model="",
    residual="",
    estimates="",
    rms=-1,
    comp_name_list=None,
    print_result=True,
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
    result = casatasks.imfit(imagename, region=region, mask=mask, model=model, residual=residual, estimates=estimates, rms=rms)
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

    # if plot:
    #     fig = plt.figure(figsize=(12, 4))

    #     # data
    #     header = fits.getheader(imagepath)
    #     data = fits.getdata(imagepath)
    #     beam = (header['BMAJ']/np.abs(header['CDELT1']), header['BMIN']/np.abs(header['CDELT2']), header['BPA'] + 90)
    #     ax = fig.add_subplot(131, projection=WCS(header))
    #     plot_2D_map(data[plot_region_slices], ax=ax, contour=False, beam=beam, title="Data", **plot_kw)

    #     # region
    #     fit_region = Regions.parse(region + ' coord=' + header['RADESYS'].lower(), format='crtf')[0]
    #     fit_region.to_pixel(WCS(header)).plot(ax=ax, facecolor="none", edgecolor="white", linestyle="dashed")

    #     # model
    #     # export to FITS and read it
    #     modfits = model + ".fits"
    #     casatasks.exportfits(model, fitsimage=modfits, overwrite=True, dropdeg=True)
    #     header = fits.getheader(modfits)
    #     data = fits.getdata(modfits)

    #     # plot
    #     ax = fig.add_subplot(132, projection=WCS(header))
    #     plot_2D_map(data[plot_region_slices], ax=ax, contour=False, title="Model", **plot_kw)

    #     # region
    #     fit_region = Regions.parse(region + ' coord=' + header['RADESYS'].lower(), format='crtf')[0]
    #     fit_region.to_pixel(WCS(header)).plot(ax=ax, facecolor="none", edgecolor="white", linestyle="dashed")
    #     # visual clarity
    #     ax.tick_params(axis="x", labelbottom=False)  # remove ticklabels for visual clarity
    #     ax.tick_params(axis="y", labelleft=False)

    #     # residual
    #     # export to FITS and read it
    #     resfits = residual + ".fits"
    #     casatasks.exportfits(residual, fitsimage=resfits, overwrite=True, dropdeg=True)
    #     header = fits.getheader(resfits)
    #     data = fits.getdata(resfits)

    #     # plot
    #     ax = fig.add_subplot(133, projection=WCS(header))
    #     plot_kw["imshow_kw"] = {
    #         "cmap": "RdBu_r",
    #         "vmin": -3 * rms,
    #         "vmax": 3 * rms,
    #     }  # change to diverging cmap and rms limited color range
    #     plot_2D_map(data[plot_region_slices], ax=ax, contour=False, title="Residual", **plot_kw)

    #     # region
    #     fit_region = Regions.parse(region + ' coord=' + header['RADESYS'].lower(), format='crtf')[0]
    #     fit_region.to_pixel(WCS(header)).plot(ax=ax, facecolor="none", edgecolor="black", linestyle="dashed")
    #     # visual clarity
    #     ax.tick_params(axis="x", labelbottom=False)  # remove ticklabels for visual clarity
    #     ax.tick_params(axis="y", labelleft=False)

    #     plt.subplots_adjust(wspace=0.4)

    return output_result