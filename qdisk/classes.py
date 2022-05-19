import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
import casatasks
import numpy as np
import astropy.constants as ac
import astropy.units as u
import matplotlib.pyplot as plt

deg_to_rad = np.pi / 180.0
deg_to_arcsec = 3600.
ckms = ac.c.to(u.km/u.s).value

class FitsImage:

    def __init__(self, fitsname, data_squeezed=False, beam_in_arcsec=True):

        self.fitsname = fitsname

        # header
        self.header = fits.getheader(fitsname)

        # read header
        self.ndim = self.header["NAXIS"]
        self.data_unit = self.header["BUNIT"]

        # data
        self.data = fits.getdata(fitsname)
        if data_squeezed:
            self.data = np.squeeze(self.data)
            self.ndim = self.data.ndim

        # get axes data
        self.get_axes()
        
        # pixel scale
        self.Omega_pix = np.abs(self.dx) * np.abs(self.dy)

        # beam
        self.get_beam_info(in_arcsec=beam_in_arcsec)

        # frequency
        if self.ndim > 2:
            self.restfreq = self.header["RESTFRQ"]

    def get_axes(self):
        # if self.ndim == 4:
        #     axes = ["x", "y", "z", "w"] # ra, dec, spectral, stokes
        if self.ndim == 3:
            axes = ["x", "y", "z"] # ra, dec, spectral
        elif self.ndim == 2:
            axes = ["x", "y"] # ra, dec
        for i, ax in enumerate(axes):
            setattr(self, "n"+ax, self.header["NAXIS{:d}".format(i+1)]) # header axis numbered from 1
            setattr(self, "d"+ax, self.header["CDELT{:d}".format(i+1)])
            setattr(self, ax+"0", self.header["CRVAL{:d}".format(i+1)])
            setattr(self, "u"+ax, self.header["CUNIT{:d}".format(i+1)])

    def get_beam_info(self, in_arcsec=False):
        """Fetch the beam information in header.

        Args:
            header (str): FITS header.

        Returns:
            tuple: beam info in units of arcsec.
        """
        ### assume in deg in header
        self.bmaj = self.header['BMAJ']
        self.bmin = self.header['BMIN']
        self.bpa = self.header['BPA']
        
        self.Omega_beam = np.pi / (4 * np.log(2)) * self.bmaj * self.bmin

        if in_arcsec:
            self.bmaj *= deg_to_arcsec
            self.bmin *= deg_to_arcsec

        self.beam = (self.bmaj, self.bmin, self.bpa)

        return self.beam # to make accesible from outside
    
    def get_directional_coord(self, center_coord=None, in_arcsec=True):
        """Generate a (RA\cos(Dec), Dec) coordinate (1D each) in arcsec. Assume the unit for coordinates in the header is deg.

        Args:
            header (dict): FITS header.
            center_coord (tuple or astropy.coordinates.SkyCoord object, optinal): Two component tuple of (RA, Dec) in arcsec or the SkyCoord object for the center coordinate. Defaults to (0.0, 0.0)

        Returns:
            tuple: Coordinates
        """

        x0 = self.x0
        y0 = self.y0
        dx = self.dx
        dy = self.dy

        if in_arcsec:
            assert self.ux == self.uy == "deg"
            x0 *= deg_to_arcsec
            y0 *= deg_to_arcsec
            dx *= deg_to_arcsec
            dy *= deg_to_arcsec
        if center_coord is None:
            offset_x, offset_y = 0, 0
        else:
            if isinstance(center_coord, tuple):
                center_x, center_y = center_coord
            elif isinstance(center_coord, SkyCoord):
                center_x = center_coord.ra.arcsec 
                center_y = center_coord.dec.arcsec
            offset_x = center_x - x0 # offset along x from phsecenter in arcsec
            offset_y = center_y - y0 # offset along y from phsecenter in arcsec

        self.x = dx * (np.arange(self.nx) - (self.header['CRPIX1'] - 1)) - offset_x
        self.y = dy * (np.arange(self.ny) - (self.header['CRPIX2'] - 1)) - offset_y

    
    def get_projected_coord(self, PA=90., incl=0., center_coord=None, in_arcsec=True): 
        self.get_directional_coord(center_coord=center_coord, in_arcsec=in_arcsec)

        # meshgrid to be in 2D
        xx, yy = np.meshgrid(self.x, self.y)

        # project to the disk plane; assume geometrically thin disk
        incl = np.radians(incl)
        PA = np.radians(PA)

        self.x_proj = (xx * np.sin(PA) + yy * np.cos(PA)) 
        self.y_proj = (- xx * np.cos(PA) + yy * np.sin(PA)) / np.cos(incl) # follow the formulation in Yen et al. 2016

        # if cart_or_pol == 'cart':
        #     return self.x_proj, self.y_proj

        # polar coordinate
        self.r = np.sqrt(self.x_proj**2 + self.y_proj**2) # in arcsec
        self.theta = np.degrees(np.arctan2(self.y_proj, self.x_proj)) # in degree, [-180, 180]

        # if cart_or_pol == 'polar':
        #     return self.r, self.theta
        
        # if cart_or_pol == 'both':
        #     return (self.x_proj, self.y_proj), (self.r, self.theta)


    def get_spectral_coord(self):
        if self.ndim < 3:
            raise KeyError("Spectral axis not found.")
        
        # assume in frequency
        self.nu = self.dz * (np.arange(self.nz) - (self.header['CRPIX3'] - 1)) + self.z0

        # if freq_or_vel == 'freq':
        #     return self.nu

        assert self.header['VELREF'] == 257 # in radio convention
        self.v = ckms * (1 - self.nu / self.restfreq)

        # if freq_or_vel == 'vel':
        #     return self.v
        
        # if freq_or_vel == 'both':
        #     return self.nu, self.v


    def get_mask(self, center_coord=None, rmin=0.0, rmax=np.inf, thetamin=-180., thetamax=180., PA=90., incl=0.0, vmin=-np.inf, vmax=np.inf):
        # get projected coordinate
        self.get_projected_coord(center_coord=center_coord, in_arcsec=True, PA=PA, incl=incl)

        # radial mask
        r_mask = np.where(((self.r >= rmin) & (self.r <= rmax)), 1.0, 0.0)

        # azimuthal mask
        theta_mask = np.where(((self.theta >= thetamin) & (self.theta <= thetamax)), 1.0, 0.0)

        # channel mask
        if self.ndim > 2:
            # have 3D directinal masks
            r_mask = np.expand_dims(r_mask, axis=0)
            theta_mask = np.expand_dims(theta_mask, axis=0)

            # calculate velocity axis
            self.get_spectral_coord()
            channel_mask = np.where(((self.v >= vmin) & (self.v <= vmax)), 1.0, 0.0)
            channel_mask = np.expand_dims(channel_mask, axis=(1,2))

        # combine
        self.mask = np.logical_and(r_mask, theta_mask)

        if self.ndim > 2:
            self.mask = np.logical_and(self.mask, channel_mask)


    def save_mask(self, maskname=None, overwrite=True, import_casa=False):
        if self.ndim > self.mask.ndim: # data dimension is 3D, whereas mask is 2D
            self.mask = np.expand_dims(self.mask, axis=0)

        # a few tweaks in the header
        # del self.header["BMAJ"]
        # del self.header["BMIN"]
        # del self.header["BPA"]
        # self.header["BTYPE"] = ""
        # self.header["BUNIT"] = ""
        self.header["COMMENT"] = "0/1 mask created by python script"

        # save as a FITS
        if maskname is None:
            maskname = self.fitsname.replace(".fits", ".mask.fits")
        fits.writeto(maskname, self.mask.astype(float), self.header, overwrite=overwrite, output_verify="silentfix")

        if import_casa:
            casatasks.importfits(fitsimage=maskname, imagename=maskname.replace(".fits", ".image"), overwrite=overwrite)

    
    def estimate_rms(self, **mask_kwargs):
        self.get_mask(**mask_kwargs)

        # masked data
        masked_data = self.data * self.mask

        # calculate rms
        self.rms = np.nanstd(masked_data[masked_data.nonzero()])









        
        


        