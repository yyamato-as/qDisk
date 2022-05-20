from astropy.coordinates import SkyCoord
from qdisk.classes import FitsImage
from qdisk.fit import imfit_wrapper
from qdisk.statistics import measure_noise_level
import eDiskplot as eplot
import os
import matplotlib.pyplot as plt
from astropy.visualization import AsinhStretch
import pickle
import casatasks

source = "L1489IRS"
baseline = "SBLB"
robust = 1.0
prefix = "{:s}_{:s}_continuum_robust_{}".format(source, baseline, robust)
imagepath = "/raid/work/yamato/eDisk_data/L1489IRS/ALMA_pipeline_reduced_data/try1_continuum_nterms1/"


imagename = imagepath + prefix + ".image.tt0.fits"

# nominal source position and noise annulus region
dir = "04h04m43.070001s +26d18m56.20011s"  # from imaging script
center_coord = SkyCoord(dir, frame="icrs")
r_in = 5  # in arcsec
r_out = 8  # in arcsec

# make directory for imfit
imfitpath = "./imfit_{:s}/".format(prefix)
os.system("mkdir " + imfitpath)

print("Loading data...")
# load the FitsImage class
image = FitsImage(imagename)

print("Measuring the rms...")
# noise mask
image.get_mask(center_coord=center_coord, rmin=r_in, rmax=r_out)

# measure the rms
rms = measure_noise_level(image.data, image.mask)

# disk mask
image.get_mask(center_coord=center_coord, rmin=0.0, rmax=r_in)

maskname = imfitpath + prefix + ".imfit.mask.fits"
image.save_mask(maskname=maskname, import_casa=True)  # assume 1/0 mask

# set the initial estimates file
# peak intensity, peak xpixel, peak ypixel, maj, min, pa
# values from Sai et al. 2020
est_str_list = [
    "0.003, 3000, 3000, 0.097arcsec, 0.037arcsec, 49deg\n",
    "0.001, 3000, 3000, 4.1arcsec, 1.2arcsec, 69deg\n",
]  # need \n

estimates_filename = imfitpath + prefix + ".imfit.estimates.txt"
with open(estimates_filename, "w") as f:
    f.writelines(est_str_list)

# set model and residual file path
mod_filename = imfitpath + prefix + ".imfit.model.image"
res_filename = imfitpath + prefix + ".imfit.residual.image"

# fit with 2D Gaussian; takes a while (~3 min)
fit = imfit_wrapper(
    imagename=imagename,
    mask='"{}">0.0'.format(maskname),
    model=mod_filename,
    residual=res_filename,
    estimates=estimates_filename,
    rms=rms,
    print_result=True,
)

# save the results
fit_filename = imfitpath + prefix + ".imfit.result.pkl"
with open(fit_filename, "wb") as f:
    pickle.dump(fit, fit_filename, protocol=pickle.HIGHEST_PROTOCOL)

# plot the results to check the reasonablility of fit
toplot = ["data", "model", "residual"]
filename = {"data": imagename, "model": mod_filename, "residual": res_filename}
cmap = {"data": "inferno", "model": "inferno", "residual": "RdBu_r"}
norm_kwargs = {
    "data": dict(stretch=AsinhStretch(a=0.02), vmin=0.0),
    "model": dict(stretch=AsinhStretch(a=0.02), vmin=0.0),
    "residual": dict(vmin=-3 * rms * 1e3, vmax=3 * rms * 1e3),
}

print("Plotting the fit...")
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

for i, val in enumerate(toplot):
    casatasks.exportfits(
        imagename=filename[val],
        fitsimage=filename[val] + ".fits",
        overwrite=True,
        dropdeg=True,
    )

    ax = axes[i]
    eplot.plot_map(
        filename[val] + ".fits",
        ax=ax,
        center_coord=center_coord,
        data_scaling_factor=1e3,
        scale=50,
        distance=140,
        unit=r"mJy beam$^{-1}$" if i == 2 else None,
        cmap=cmap[val],
        norm_kwargs=norm_kwargs[val],
    )

    axes[i].set(
        xlim=(r_in, -r_in),
        ylim=(-r_in, r_in),
        xlabel=r"$\Delta$R.A. [arcsec]",
        title=val
    )
    if i == 0:
        ax.set(
            ylabel=r"$\Delta$Dec [arcsec]",
        )

fig.savefig(
    "./imfit_L1489IRS_SBLB_continuum_robust_1.0/L1489IRS_SBLB_continuum_robust_1.0.imfit.fit.png",
    bbox_inches="tight",
    pad_inches=0.01,
    dpi=500
)

# clean up
os.system("rm -r " + imfitpath + "*.image")
