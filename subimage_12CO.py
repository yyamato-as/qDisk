# trim the 12CO image cube for relevant velocity range to make the cube easier to handle
import sys
import casatasks
import numpy as np
from qdisk.data_product import get_cor_channel
from qdisk.classes import FitsImage
import subprocess

source = sys.argv[1]
molecular_species = ["12CO"]#, "13CO", "C18O", "SO"]
baseline = ["SB", "SBLB"]
v_range = (-12.0, 28.0)

for ms in molecular_species:
    for bl in baseline:

        imagename = "/raid/work/yamato/eDisk_data/L1489IRS/ALMA_pipeline_reduced_data/try1_continuum_nterms1/{:s}_{:s}_{:s}_robust_0.5.image.fits".format(
            source, bl, ms
        )

        image = FitsImage(fitsname=imagename)
        image.get_spectral_coord()

        chans = "~".join([str(np.argmin(np.abs(image.v - v))) for v in v_range])
        print("Chosen channels: " + chans)

        print("Processing {:s}...".format(imagename))
        outfile = imagename.replace(".fits", ".sub")
        subprocess.run(["rm", "-r", outfile])
        casatasks.imsubimage(imagename=imagename, outfile=outfile, chans=chans)
        print("Exporting into fits...")
        casatasks.exportfits(
            imagename=outfile,
            fitsimage=outfile + ".fits",
            overwrite=True,
            dropdeg=True,
        )
