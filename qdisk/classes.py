import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.constants as ac
import astropy.units as u
import matplotlib.pyplot as plt
from .utils import is_within
import casatools, casatasks
from qdisk.utils import remove_casalogfile
remove_casalogfile()

deg_to_rad = np.pi / 180.0
deg_to_arcsec = 3600.0
rad_to_arcsec = 180.0 / np.pi * 3600.0
ckms = ac.c.to(u.km / u.s).value

axes_attr = {"Right Ascension": "x", "Declination": "y", "Stokes": "w", "Frequency": "z"}

ia = casatools.image()

class CasaImage:

    def __init__(self, imagename):

        self.fitsname = imagename

        # header
        self.header = casatasks.imhead(imagename, mode="list")

        # read header
        self.ndim = len(self.header["shape"])
        self.data_unit = self.header["bunit"]

        # data
        ia.fromimage(infile=imagename)
        self.data = ia.torecord()["imagearray"].T
        # if data_squeezed:
        #     self.data = np.squeeze(self.data)
        #     self.ndim = self.data.ndim

        # get axes data
        self.get_axes()
        
        # pixel scale
        self.Omega_pix = np.abs(self.dx) * np.abs(self.dy)

        # beam
        self.get_beam_info()

        # frequency
        if self.ndim > 2:
            self.restfreq = self.header["restfreq"]

    def get_axes(self):
        axes = [axes_attr[self.header["ctype{:d}".format(i+1)]] for i in range(self.ndim)]
        for i, ax in enumerate(axes):
            setattr(self, "n"+ax, self.data.shape[i]) # header axis numbered from 1
            setattr(self, "d"+ax, self.header["cdelt{:d}".format(i+1)])
            setattr(self, ax+"0", self.header["crval{:d}".format(i+1)])
            setattr(self, "u"+ax, self.header["cunit{:d}".format(i+1)])
            setattr(self, "r"+ax, self.header["crpix{:d}".format(i+1)])

    def get_beam_info(self):
        """Fetch the beam information in header.

        Args:
            header (str): FITS header.

        Returns:
            tuple: beam info in units of arcsec.
        """
        ### assume in deg in header
        self.bmaj = self.header['beammajor']["value"]
        self.bmin = self.header['beamminor']["value"]
        self.bpa = self.header['beampa']["value"]
        
        self.Omega_beam = np.pi / (4 * np.log(2)) * self.bmaj * self.bmin

        if self.header['beammajor']["unit"] == "deg":
            self.bmaj *= deg_to_arcsec
            self.bmin *= deg_to_arcsec

        self.beam = (self.bmaj, self.bmin, self.bpa)

        return self.beam # to make accesible from outside
    
    def get_directional_coord(self, center_coord=None):
        """Generate a (RA\cos(Dec), Dec) coordinate (1D each) in arcsec. Assume the unit for coordinates in the header is deg.

        Args:
            header (dict): FITS header.
            center_coord (tuple or astropy.coordinates.SkyCoord object, optinal): Two component tuple of (RA, Dec) in arcsec or the SkyCoord object for the center coordinate. Defaults to (0.0, 0.0)

        Returns:
            tuple: Coordinates
        """

        assert self.ux == self.uy == "rad"
        x0 = self.x0 * rad_to_arcsec
        y0 = self.y0 * rad_to_arcsec
        dx = self.dx * rad_to_arcsec
        dy = self.dy * rad_to_arcsec

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

        self.x = dx * (np.arange(self.nx) - (self.rx - 1)) - offset_x
        self.y = dy * (np.arange(self.ny) - (self.ry - 1)) - offset_y

    
    def get_projected_coord(self, PA=90., incl=0., center_coord=None): 
        self.get_directional_coord(center_coord=center_coord)

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
        self.nu = self.dz * (np.arange(self.nz) - (self.rz - 1)) + self.z0

        # if freq_or_vel == 'freq':
        #     return self.nu

        # assert self.header['VELREF'] == 257 # in radio convention
        self.v = ckms * (1 - self.nu / self.restfreq)

        # if freq_or_vel == 'vel':
        #     return self.v
        
        # if freq_or_vel == 'both':
        #     return self.nu, self.v

    @staticmethod
    def calc_inclination(maj, min):
        return np.rad2deg(np.arccos(min / maj)) % 360

    def get_beam_mask(self, center_coord=None):
        beam_incl = self.calc_inclination(self.bmaj, self.bmin)
        
        self.get_projected_coord(center_coord=center_coord, PA=self.bpa, incl=beam_incl)

        self.beam_mask = np.where(self.r <= self.bmaj, 1.0, 0.0)

    
    def get_spectrum(self, center_coord=None):
        if self.ndim < 3:
            raise ValueError("The image is 2D. Can't get the spectrum.")

        self.get_spectral_coord()

        self.get_beam_mask(center_coord=center_coord)

        masked_data = self.data.squeeze().T * self.beam_mask

        self.spec = np.array([np.nanmean(im[im.nonzero()]) for im in masked_data])

    def get_mask(self, center_coord=None, rmin=0.0, rmax=np.inf, thetamin=-180., thetamax=180., PA=90., incl=0.0, vmin=-np.inf, vmax=np.inf):
        # get projected coordinate
        self.get_projected_coord(center_coord=center_coord, PA=PA, incl=incl)

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


class FitsImage:
    def __init__(self, fitsname):

        self.fitsname = fitsname

        # header
        self.header = fits.getheader(fitsname)

        # read header
        self.ndim = self.header["NAXIS"]
        self.data_unit = self.header["BUNIT"]

        # data
        self.data = fits.getdata(fitsname)

        # get axes data
        self._get_axes_info()

        # pixel sold angle
        self.Omega_pix = np.abs(self.dx) * np.abs(self.dy)

        # beam
        self._get_beam_info()

        # frequency
        if "RESTFRQ" in self.header:
            self.restfreq = self.header["RESTFRQ"]

    def _get_axes_info(self):
        ### standard image (2D or 3D) should have two directinal (and one spectral) axes;
        ### or, non-standard (e.g., PV diagram) should have one directinal and one spectral axes.
        ### This will be dealt with by "CTYPEi" argument in the header

        if self.ndim == 4:
            axes = ["x", "y", "z", "w"]  # ra, dec, spectral, stokes
        elif self.ndim == 3:
            axes = ["x", "y", "z"]  # ra, dec, spectral; or directinal, spectral, stokes
        elif self.ndim == 2:
            axes = ["x", "y"]  # ra, dec; or directinal, spectral
        for i, ax in enumerate(axes):
            setattr(
                self, "n" + ax, self.header["NAXIS{:d}".format(i + 1)]
            )  # header axis numbered from 1
            setattr(self, "d" + ax, self.header["CDELT{:d}".format(i + 1)])
            setattr(self, ax + "0", self.header["CRVAL{:d}".format(i + 1)])
            setattr(self, "u" + ax, self.header["CUNIT{:d}".format(i + 1)])

    def _get_beam_info(self):
        """Fetch the beam information in header.

        Args:
            header (str): FITS header.

        Returns:
            tuple: beam info in units of arcsec.
        """
        ### assume in deg in header
        self.bmaj = self.header["BMAJ"] * deg_to_arcsec
        self.bmin = self.header["BMIN"] * deg_to_arcsec
        self.bpa = self.header["BPA"]

        ### beam solid angle
        self.Omega_beam = np.pi / (4 * np.log(2)) * self.bmaj * self.bmin

        ### tuple of bmaj, bmin, and bpa
        self.beam = (self.bmaj, self.bmin, self.bpa)

        # return self.beam # to make accesible from outside

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
            elif isinstance(center_coord, str):
                center_coord = SkyCoord(center_coord, frame="icrs")
                center_x = center_coord.ra.arcsec
                center_y = center_coord.dec.arcsec
            offset_x = center_x - x0  # offset along x from phsecenter in arcsec
            offset_y = center_y - y0  # offset along y from phsecenter in arcsec

        self.x = dx * (np.arange(self.nx) - (self.header["CRPIX1"] - 1)) - offset_x
        self.y = dy * (np.arange(self.ny) - (self.header["CRPIX2"] - 1)) - offset_y

    def get_projected_coord(self, PA=90.0, incl=0.0, center_coord=None, in_arcsec=True):
        self.get_directional_coord(center_coord=center_coord, in_arcsec=in_arcsec)

        # meshgrid to be in 2D
        xx, yy = np.meshgrid(self.x, self.y)

        # project to the disk plane; assume geometrically thin disk
        incl = np.radians(incl)
        PA = np.radians(PA)

        self.x_proj = xx * np.sin(PA) + yy * np.cos(PA)
        self.y_proj = (-xx * np.cos(PA) + yy * np.sin(PA)) / np.cos(
            incl
        )  # follow the formulation in Yen et al. 2016

        # polar coordinate
        self.r = np.sqrt(self.x_proj**2 + self.y_proj**2)  # in arcsec
        self.theta = np.degrees(
            np.arctan2(self.y_proj, self.x_proj)
        )  # in degree, [-180, 180]

    def get_spectral_coord(self):
        if self.ndim < 3:
            raise KeyError("Spectral axis not found.")

        # assume in frequency
        self.nu = self.dz * (np.arange(self.nz) - (self.header["CRPIX3"] - 1)) + self.z0

        assert self.header["VELREF"] == 257  # in radio convention
        self.v = ckms * (1 - self.nu / self.restfreq)

    
    def cutout(self, xlim=None, ylim=None, vlim=None):
        if xlim is not None:
            self.data = self.data[..., is_within(self.x, xlim)]
            try:
                self.x_proj = self.x_proj[:, is_within(self.x, xlim)]
                self.y_proj = self.y_proj[:, is_within(self.x, xlim)]
                self.r = self.r[:, is_within(self.x, xlim)]
                self.theta = self.theta[:, is_within(self.x, xlim)]
            except AttributeError:
                pass
            self.x = self.x[is_within(self.x, xlim)]

        if ylim is not None:
            self.data = self.data[..., is_within(self.y, ylim), :]
            try:
                self.x_proj = self.x_proj[is_within(self.y, ylim), :]
                self.y_proj = self.y_proj[is_within(self.y, ylim), :]
                self.r = self.r[is_within(self.y, ylim), :]
                self.theta = self.theta[is_within(self.y, ylim), :]
            except AttributeError:
                pass
            self.y = self.y[is_within(self.y, ylim)]

        if vlim is not None:
            self.data = self.data[is_within(self.v, vlim)]
            self.nu = self.nu[is_within(self.v, vlim)]
            self.v = self.v[is_within(self.v, vlim)]
        
    def downsample(self, N):
        # adopted from eddy code by rich teague
        N = int(np.ceil(self.bmaj / (abs(self.dx)*deg_to_arcsec))) if N == 'beam' else N
        N0x, N0y = int(N / 2), int(N / 2)
        if N > 1:
            self.x = self.x[N0x::N]
            self.y = self.y[N0y::N]
            self.data = self.data[..., N0y::N, N0x::N]
            try:
                self.x_proj = self.x_proj[N0y::N, N0x::N]
                self.y_proj = self.y_proj[N0y::N, N0x::N]
                self.r = self.r[N0y::N, N0x::N]
                self.theta = self.theta[N0y::N, N0x::N]
                # self.mask = self.mask[N0y::N, N0x::N]
            except AttributeError:
                pass
            # self.error = self.error[N0y::N, N0x::N]
            # self.mask = self.mask[N0y::N, N0x::N]

    def get_threshold_mask(self, threshold=3):
        self.SNR = self.data / self.rms 
        self.threshold_mask = self.SNR >= threshold

    def get_mask(
        self,
        center_coord=None,
        rmin=0.0,
        rmax=np.inf,
        thetamin=-180.0,
        thetamax=180.0,
        PA=90.0,
        incl=0.0,
        vmin=-np.inf,
        vmax=np.inf,
    ):
        # get projected coordinate
        self.get_projected_coord(
            center_coord=center_coord, in_arcsec=True, PA=PA, incl=incl
        )

        # radial mask
        r_mask = np.where(((self.r >= rmin) & (self.r <= rmax)), 1.0, 0.0)

        # azimuthal mask
        theta_mask = np.where(
            ((self.theta >= thetamin) & (self.theta <= thetamax)), 1.0, 0.0
        )

        # channel mask
        if self.ndim > 2:
            # have 3D directinal masks
            r_mask = np.expand_dims(r_mask, axis=0)
            theta_mask = np.expand_dims(theta_mask, axis=0)

            # calculate velocity axis
            self.get_spectral_coord()
            channel_mask = np.where(((self.v >= vmin) & (self.v <= vmax)), 1.0, 0.0)
            channel_mask = np.expand_dims(channel_mask, axis=(1, 2))

        # combine
        self.mask = np.logical_and(r_mask, theta_mask)

        if self.ndim > 2:
            self.mask = np.logical_and(self.mask, channel_mask)

    def save_mask(self, maskname=None, overwrite=True, import_casa=False):
        if self.ndim > self.mask.ndim:  # data dimension is 3D, whereas mask is 2D
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
        fits.writeto(
            maskname,
            self.mask.astype(float),
            self.header,
            overwrite=overwrite,
            output_verify="silentfix",
        )

        if import_casa:
            import casatasks
            from .utils import remove_casalogfile

            remove_casalogfile()

            casatasks.importfits(
                fitsimage=maskname,
                imagename=maskname.replace(".fits", ".image"),
                overwrite=overwrite,
            )

    def estimate_rms(self, edgenchan=None, **mask_kwargs):

        if edgenchan is not None:
            self.rms = np.nanstd([self.data[:edgenchan], self.data[-edgenchan:]])

        else:
            self.get_mask(**mask_kwargs)
            self.rms = np.nanstd(self.data[self.mask != 0.0])

    
    def spectrally_collapse(self, vrange=None, sigma_clip=None, rms=None, noiseedgenchan=3, mode="average"):

        if self.ndim < 3:
            raise ValueError("The image is not 3D. Spectral collapse is not avilable.")

        if rms is None:
            self.estimate_rms(edgenchan=noiseedgenchan)
            rms = self.rms
        
        data = self.data

        self.get_spectral_coord()
        v = self.v

        if vrange is not None:
            data = data[is_within(self.v, vrange),:,:]
            v = v[is_within(self.v, vrange)]

        if sigma_clip is not None:
            data[data < sigma_clip * rms] = np.nan

        # collapse
        if "ave" in mode:
            self.collapsed = np.nanaverage(data, axis=0)
        elif "s" in mode:
            self.collapsed = np.nansum(data, axis=0)
        elif ("mom0" in mode) or ("integ" in mode):
            dchan = np.diff(v).mean()
            data[np.isnan(data)] = 0.0
            self.collapsed = np.trapz(data, dx=dchan, axis=0)




class PVFitsImage(FitsImage):
    def __init__(self, fitsname):
        super().__init__(fitsname)
        self._get_PVaxes_info()

    def _get_PVaxes_info(self):
        for i in range(self.ndim):
            if "offset" in self.header["CTYPE{:d}".format(i + 1)].lower():
                # read off position axes info
                self.np = self.header["NAXIS{:d}".format(i + 1)]  # number of pixels
                self.dp = self.header["CDELT{:d}".format(i + 1)]  # increment
                self.p0 = self.header["CRVAL{:d}".format(i + 1)]  # reference value
                self.rp = self.header["CRPIX{:d}".format(i + 1)]  # reference pixel
                self.up = self.header["CUNIT{:d}".format(i + 1)]  # unit

            if "freq" in self.header["CTYPE{:d}".format(i + 1)].lower():
                # read off velocity axis info in the case header is in frequency
                self.nv = self.header["NAXIS{:d}".format(i + 1)]  # number of pixels
                self.dv = (
                    self.header["CDELT{:d}".format(i + 1)] * ckms / self.restfreq
                )  # increment
                self.v0 = ckms * (
                    1.0 - self.header["CRVAL{:d}".format(i + 1)] / self.restfreq
                )  # reference value
                self.rv = self.header["CRPIX{:d}".format(i + 1)]  # reference pixel
                self.uv = "km/s"  # unit

            elif "velocity" in self.header["CTYPE{:d}".format(i + 1)].lower():
                # read off velocityaxis info in the case header is in velocity
                self.nv = self.header["NAXIS{:d}".format(i + 1)]  # number of pixels
                self.dv = self.header["CDELT{:d}".format(i + 1)]  # increment
                self.v0 = self.header["CRVAL{:d}".format(i + 1)]  # reference value
                self.rv = self.header["CRPIX{:d}".format(i + 1)]  # reference pixel
                self.uv = self.header["CUNIT{:d}".format(i + 1)]  # unit

            else:
                continue

    def get_coord(self):
        # fetch all coordinates
        self.get_position_coord()
        self.get_velocity_coord()

    def get_position_coord(self):
        self.p = self.dp * (np.arange(self.np) - (self.rp - 1)) + self.p0
        # unit conversion
        if self.up == "deg":
            self.p *= deg_to_arcsec
        if self.up == "rad":
            self.p *= rad_to_arcsec

    def get_velocity_coord(self):
        self.v = self.dv * (np.arange(self.nv) - (self.rv - 1)) + self.v0
