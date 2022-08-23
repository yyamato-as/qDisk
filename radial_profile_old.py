from qdisk import product as dp
from eDisk_source_dict import source_dict
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
import sys

source = sys.argv[1]
molecular_lines = [
    # "12CO",
    "13CO",
    "C18O",
    "SO",
    # "DCN",
    # "CH3OH",
    # "H2CO_3_03-2_02_218.22GHz",
    # "H2CO_3_21-2_20_218.76GHz",
    # "H2CO_3_22-2_21_218.47GHz",
    # "SiO",
    # "c-C3H2_217.82",
    # "cC3H2_217.94",
    # "cC3H2_218.16",
]

baselines = ["SBLB"]

imagepath = "/raid/work/yamato/eDisk_data/L1489IRS/ALMA_pipeline_reduced_data/try1_continuum_nterms1/"
savefilepath = "/raid/work/yamato/eDisk_data/L1489IRS/data_product_test/"
wedges = [30, 60, 90, 120, 150, 180]
rmax = 3

# continuum
for bl in baselines:
    for wedge in wedges:
        print("Start processing continuum {:s}...".format(bl))
        imagename = imagepath + "{:s}_{:s}_continuum_robust_1.0.image.tt0.fits".format(
            source, bl
        )

        savefilename = (
            savefilepath
            + "{:s}_{:s}_continuum_robust_1.0.profile.wedge{}.txt".format(
                source, bl, wedge
            )
        )

        x, y, dy = dp.calculate_radial_profile(
            imagename,
            PA=source_dict[source]["PA"],
            incl=source_dict[source]["incl"],
            center_coord=SkyCoord(source_dict[source]["radec"]),
            rmax=rmax,
            wedge_angle=wedge,
        )

        tosave = np.stack((x, y, dy), axis=-1)

        np.savetxt(savefilename, tosave, header="r [arcsec] \t I [Jy/beam] \t dI [Jy/beam]")


for line in molecular_lines:
    for bl in baselines:
        for wedge in wedges:
            print("Start processing {:s} {:s}...".format(line, bl))
            imagename = imagepath + "{:s}_{:s}_{:s}_robust_0.5.image_M0.fits".format(
                source, bl, line
            )
            if line == "12CO":
                imagepath = imagepath.replace(".image_M0.fits", ".image.sub_M0.fits")

            savefilename = (
                savefilepath
                + "{:s}_{:s}_{:s}_robust_0.5.profile.wedge{}.txt".format(
                    source, bl, line, wedge
                )
            )

            x, y, dy = dp.calculate_radial_profile(
                imagename,
                PA=source_dict[source]["PA"],
                incl=source_dict[source]["incl"],
                center_coord=SkyCoord(source_dict[source]["radec"]),
                rmax=rmax,
                wedge_angle=wedge,
            )

            tosave = np.stack((x, y, dy), axis=-1)

            np.savetxt(savefilename, tosave, header="r [arcsec] \t I [mJy/beam km/s] \t dI [mJy/beam km/s]")

