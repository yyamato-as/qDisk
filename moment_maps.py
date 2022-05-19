from qdisk import data_product as dp
import numpy as np
from eDisk_source_dict import source_dict
import sys

source = sys.argv[1]
molecular_lines = [
    "12CO",
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
molecular_lines = ["SO"]
moments = [0, 1, 8]
baselines = ["SB", "SBLB"]

imagepath = "/raid/work/yamato/eDisk_data/L1489IRS/ALMA_pipeline_reduced_data/try1_continuum_nterms1/"
savefilepath = "/raid/work/yamato/eDisk_data/L1489IRS/data_product_test/"

for line in molecular_lines:
    for bl in baselines:
        imagename =  imagepath + "{:s}_{:s}_{:s}_robust_0.5.image.fits".format(
            source, bl, line
        )
        if line == "12CO":
            imagename = imagename.replace(".fits", ".sub.fits")
        savefilename = savefilepath + "{:s}_{:s}_{:s}_robust_0.5.image.fits".format(
            source, bl, line
        )
        kwargs = {
            0: dict(),
            1: dict(),
            8: dict(),
        }
        print("Start processing {}...".format(line))
        for mom in moments:
            dp.calculate_moment(
                imagename=imagename,
                moments=[mom],
                verbose=True,
                nchunks=4 if bl == "SBLB" else None,
                savefilename=savefilename,
                **kwargs[mom]
            )
