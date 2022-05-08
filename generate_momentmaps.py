from sys import argv
from qdisk import data_product as dp
import numpy as np
import sys

source = sys.argv[1]
molecular_lines = [
    "12CO",
    "13CO",
    "C18O",
    "SO",
    "DCN",
    "CH3OH",
    "H2CO_3_03-2_02_218.22GHz",
    "H2CO_3_21-2_20_218.76GHz",
    "H2CO_3_22-2_21_218.47GHz",
    "SiO",
    "c-C3H2_217.82",
    "cC3H2_217.94",
    "cC3H2_218.16",
]
# molecular_lines = ["12CO"]
# molecular_lines = ["13CO", "SiO"]
moments = [0, 1, 8]
threshold = (-np.inf, 3)

for line in molecular_lines:
    imagepath = "/raid/work/yamato/eDisk_data/L1489IRS/ALMA_pipeline_reduced_data/try1_continuum_nterms1/{:s}_SBLB_{:s}_robust_0.5.image.fits".format(
        source, line
    )
    if line == "12CO":
        imagepath = imagepath.replace(".fits", ".sub.fits")
    savefilepath = imagepath.replace(".fits", "")
    print("Start processing {}...".format(line))
    dp.calculate_moment(
        imagepath=imagepath, moments=moments, verbose=True, threshold=threshold, nchunks=4
    )  # , vel_extent=source_dict[source]["emission_extent"][line])
