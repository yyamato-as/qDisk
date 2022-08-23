from qdisk.classes import FitsImage
from eDisk_source_dict import source_dict
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



