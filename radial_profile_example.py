from qdisk.classes import FitsImage
from qdisk.plot import Map
from eDisk_source_dict import source_dict
from astropy.visualization import AsinhStretch
import matplotlib.pyplot as plt

source = "L1489IRS"
config = "SBLB"
center_coord = source_dict[source]["radec"]
imagepath = "/works/yamato/eDisk/L1489IRS/custom_images/"
wedge_angle = 45
rmax = 8

imagename = imagepath + "{:s}_{:s}_continuum_robust_1.0.image.tt0.fits".format(
            source, config
        )

image = FitsImage(imagename, xlim=(-rmax, rmax), ylim=(-rmax, rmax))
image.shift_phasecenter_toward(center_coord)
r, I, dI = image.radial_profile(PA=source_dict[source]["PA"], incl=source_dict[source]["incl"], rmax=rmax, wedge_angle=wedge_angle, save=False)

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(r, I)
ax.fill_between(r, I-dI, I+dI, alpha=0.25)
ax.set(xlim=(0.05, rmax), ylim=(0.02e-3, 20e-3), xlabel="Radius [arcsec]", ylabel="$I$ [Jy / beam]", xscale="log", yscale="log")

plt.show()

# check the average region
fig, ax = plt.subplots(constrained_layout=True)
imagemap = Map(imagename, ax=ax, center_coord=center_coord, xlim=(-rmax, rmax), ylim=(-rmax, rmax))
imagemap.plot_colormap(cmap="inferno", vmin=0.0, stretch=AsinhStretch(a=0.02))
mask = imagemap.get_mask(rmax=rmax, thetamin=wedge_angle, thetamax=180 - wedge_angle, abs_theta=True, exclude_theta=True, PA=source_dict[source]["PA"], incl=source_dict[source]["incl"])
imagemap.overlay_contour(mask, x=imagemap.x, y=imagemap.y)

plt.show()



