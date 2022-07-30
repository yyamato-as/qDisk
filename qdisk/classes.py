import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp2d
import numpy as np
import astropy.constants as ac
import astropy.units as u
import matplotlib.pyplot as plt
from .utils import is_within
import casatools, casatasks
from qdisk.utils import remove_casalogfile, jypb_to_K_RJ, jypb_to_K

remove_casalogfile()

deg_to_rad = np.pi / 180
deg_to_arcsec = 3600
rad_to_arcsec = 180 / np.pi * 3600
arcsec_to_rad = np.pi / 3600 / 180
ckms = ac.c.to(u.km / u.s).value

axes_attr = {
    "Right Ascension": "x",
    "Declination": "y",
    "Stokes": "w",
    "Frequency": "z",
}

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
        axes = [
            axes_attr[self.header["ctype{:d}".format(i + 1)]] for i in range(self.ndim)
        ]
        for i, ax in enumerate(axes):
            setattr(self, "n" + ax, self.data.shape[i])  # header axis numbered from 1
            setattr(self, "d" + ax, self.header["cdelt{:d}".format(i + 1)])
            setattr(self, ax + "0", self.header["crval{:d}".format(i + 1)])
            setattr(self, "u" + ax, self.header["cunit{:d}".format(i + 1)])
            setattr(self, "r" + ax, self.header["crpix{:d}".format(i + 1)])

    def get_beam_info(self):
        """Fetch the beam information in header.

        Args:
            header (str): FITS header.

        Returns:
            tuple: beam info in units of arcsec.
        """
        ### assume in deg in header
        self.bmaj = self.header["beammajor"]["value"]
        self.bmin = self.header["beamminor"]["value"]
        self.bpa = self.header["beampa"]["value"]

        self.Omega_beam = np.pi / (4 * np.log(2)) * self.bmaj * self.bmin

        if self.header["beammajor"]["unit"] == "deg":
            self.bmaj *= deg_to_arcsec
            self.bmin *= deg_to_arcsec

        self.beam = (self.bmaj, self.bmin, self.bpa)

        return self.beam  # to make accesible from outside

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
            offset_x = center_x - x0  # offset along x from phsecenter in arcsec
            offset_y = center_y - y0  # offset along y from phsecenter in arcsec

        self.x = dx * (np.arange(self.nx) - (self.rx - 1)) - offset_x
        self.y = dy * (np.arange(self.ny) - (self.ry - 1)) - offset_y

    def get_projected_coord(self, PA=90.0, incl=0.0, center_coord=None):
        self.get_directional_coord(center_coord=center_coord)

        # meshgrid to be in 2D
        xx, yy = np.meshgrid(self.x, self.y)

        # project to the disk plane; assume geometrically thin disk
        incl = np.radians(incl)
        PA = np.radians(PA)

        self.x_proj = xx * np.sin(PA) + yy * np.cos(PA)
        self.y_proj = (-xx * np.cos(PA) + yy * np.sin(PA)) / np.cos(
            incl
        )  # follow the formulation in Yen et al. 2016

        # if cart_or_pol == 'cart':
        #     return self.x_proj, self.y_proj

        # polar coordinate
        self.r = np.sqrt(self.x_proj**2 + self.y_proj**2)  # in arcsec
        self.theta = np.degrees(
            np.arctan2(self.y_proj, self.x_proj)
        )  # in degree, [-180, 180]

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
        self.get_projected_coord(center_coord=center_coord, PA=PA, incl=incl)

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
            casatasks.importfits(
                fitsimage=maskname,
                imagename=maskname.replace(".fits", ".image"),
                overwrite=overwrite,
            )

    def estimate_rms(self, **mask_kwargs):
        self.get_mask(**mask_kwargs)

        # masked data
        masked_data = self.data * self.mask

        # calculate rms
        self.rms = np.nanstd(masked_data[masked_data.nonzero()])


class FitsImage:
    def __init__(
        self,
        fitsname,
        squeeze=True,
        rel_dir_ax=True,
        xlim=None,
        ylim=None,
        vlim=None,
        downsample=False,
    ):

        self.fitsname = fitsname

        # header
        self.header = fits.getheader(fitsname)

        # read header
        # self.ndim = self.header["NAXIS"]
        # self.data_unit = self.header["BUNIT"]

        # data
        self.data = fits.getdata(fitsname)
        if squeeze:
            self.data = np.squeeze(self.data)

        self.rel_dir_ax = rel_dir_ax

        self._get_FITS_properties(rel_dir_ax=self.rel_dir_ax)

        self.xlim = xlim
        self.ylim = ylim
        self.vlim = vlim
        self._cutout(xlim=xlim, ylim=ylim, vlim=vlim)

        self.downsample = downsample
        if self.downsample:
            if not isinstance(self.downsample, tuple):
                self.downsample = (self.downsample, 1)
            self.downsample_cube(*self.downsample)

    def _get_axis(self, n):
        npix = self.header["naxis{:d}".format(n)]
        delta = self.header["cdelt{:d}".format(n)]
        rpix = self.header["crpix{:d}".format(n)]
        rval = self.header["crval{:d}".format(n)]

        axis = delta * (np.arange(npix) - rpix + 1) + rval

        return axis

    def _get_directional_axis(self, relative=True):
        x = self._get_axis(n=1)
        y = self._get_axis(n=2)
        if relative:
            x -= self.header["crval1"]
            y -= self.header["crval2"]
        x *= deg_to_arcsec
        y *= deg_to_arcsec
        return x, y

    def _get_spectral_axis(self):
        n = self._find_spectral_axis_in_header()
        if "freq" in self.header["ctype{:d}".format(n)].lower():
            nu = self._get_axis(n=n)
            v = (self.nu0 - nu) * ckms / self.nu0
            return nu, v
        else:
            v = self._get_axis(n=n)
            vunit = self._get_axis_unit_in_header(n=n)
            if not "k" in vunit:  # the case for in m/s
                v *= 1e3  # in km/s
            nu = (1 - v / ckms) * self.nu0
        return nu, v

    def _get_axis_unit_in_header(self, n):
        return self.header["cunit{:d}".format(n)]

    def _find_spectral_axis_in_header(self):
        for n in range(1, self.header["naxis"] + 1):
            if "freq" in self.header["ctype{:d}".format(n)].lower():
                return n
            if "vel" in self.header["ctype{:d}".format(n)].lower():
                return n

    def _get_restfreq(self):
        try:
            return self.header["restfreq"]
        except KeyError:
            try:
                return self.header["restfrq"]
            except KeyError:
                if self.ndim > 2:
                    print("Warning: no rest frequency is found in header.")
                return np.nan

    def _get_FITS_properties(self, rel_dir_ax=True):

        # data unit
        self.data_unit = self.header["bunit"]

        # directional axis
        self.x, self.y = self._get_directional_axis(relative=rel_dir_ax)

        # spectral axis
        self.nu0 = self._get_restfreq()
        if self.ndim > 2:
            self.nu, self.v = self._get_spectral_axis()

        # beam
        self._get_beam_info()

    def convert_unit(self, unit="K", nu=None, RJ_approx=False):
        ### assume the original data unit is Jy / beam or mJy / beam
        if not np.isnan(self.restfreq):
            nu = self.restfreq
        elif nu is None:
            raise ValueError("Rest frequency not found. Please provide it via *nu* argument.")

        data = self.data 
        if "mJy" in self.data_unit:
            data *= 1e-3 # in Jy /beam
        
        if unit == "K":
            if RJ_approx:
                self.data = jypb_to_K_RJ(data, nu, self.beam[:2])
            else:
                self.data = jypb_to_K(data, nu, self.beam[:2])

    def shift_phasecenter(self, dx=0.0, dy=0.0):
        self.x -= dx
        self.y -= dy

    def shift_phasecenter_toward(self, coord, fix_FOV=False):
        # restore original cube
        if not fix_FOV:
            self.restore_original_cube()

        c = SkyCoord(coord, frame="icrs")
        if self.rel_dir_ax:
            dx = c.ra.arcsec - self.header["crval1"] * deg_to_arcsec
            dy = c.dec.arcsec - self.header["crval2"] * deg_to_arcsec
        else:
            x0, y0 = self.get_phasecenter_coord()
            dx = c.ra.arcsec - x0
            dy = c.dec.arcsec - y0
        self.x -= dx
        self.y -= dy

        if not fix_FOV:
            self._cutout(xlim=self.xlim, ylim=self.ylim, vlim=self.vlim)

            if self.downsample:
                if not isinstance(self.downsample, tuple):
                    self.downsample = (self.downsample, 1)
                self.downsample_cube(*self.downsample)

    def get_phasecenter_coord(self):
        x, y = self._get_directional_axis(relative=False)
        ret = []
        for c in [x, y]:
            q, mod = divmod(c.size, 2)
            if mod == 0:
                ret.append((c[q - 1] + c[q]) / 2)
            else:
                ret.append(c[q])
        return tuple(ret)

    def _cutout(self, xlim=None, ylim=None, vlim=None):
        if xlim is not None:
            self.data = self.data[..., is_within(self.x, xlim)]
            self.x = self.x[is_within(self.x, xlim)]
        if ylim is not None:
            self.data = self.data[..., is_within(self.y, ylim), :]
            self.y = self.y[is_within(self.y, ylim)]
        if vlim is not None:
            assert self.ndim > 2
            self.data = self.data[is_within(self.v, vlim), :, :]
            self.v = self.v[is_within(self.v, vlim)]

    def _downsample_spatial(self, N):
        # adopted from eddy code by rich teague
        N = int(np.ceil(self.bmaj / abs(self.dpix))) if N == "beam" else N
        if N > 1:
            self.x = self.x[::N]
            self.y = self.y[::N]
            self.data = self.data[..., ::N, ::N]

    def _downsample_spectral(self, N):
        if self.ndim < 3 and N > 1:
            raise ValueError("Spectral downsample is not available for 2D image.")
        if N > 1:
            self.v = self.v[::N]
            self.data = self.data[::N, :, :]

    def downsample_cube(self, Nxy, Nv):
        self._downsample_spatial(Nxy)
        self._downsample_spectral(Nv)

    def restore_original_cube(self, rel_dir_ax=True, squeeze=True):
        self.data = fits.getdata(self.fitsname)
        if squeeze:
            self.data = np.squeeze(self.data)

        self.rel_dir_ax = rel_dir_ax
        self._get_FITS_properties(rel_dir_ax=rel_dir_ax)

    def get_disk_coord(self, x0=0.0, y0=0.0, PA=0.0, incl=0.0, frame="polar"):
        if frame == "polar":
            return self._get_polar_disk_coord(x0=x0, y0=y0, PA=PA, incl=incl)
        if frame == "cartesian":
            return self._get_cart_disk_coord(x0=x0, y0=y0, PA=PA, incl=incl)

    def _get_cart_sky_coord(self, x0, y0):
        return np.meshgrid(self.x - x0, self.y - y0)

    def _get_cart_disk_coord(self, x0, y0, PA, incl):
        x_sky, y_sky = self._get_cart_sky_coord(x0, y0)
        x_rot, y_rot = self._rotate_coord(x_sky, y_sky, PA)
        return self._deproject_coord(x_rot, y_rot, incl)

    def _get_polar_disk_coord(self, x0, y0, PA, incl):
        x_cart, y_cart = self._get_cart_disk_coord(x0, y0, PA, incl)
        return np.hypot(x_cart, y_cart), np.arctan2(y_cart, x_cart)

    @staticmethod
    def _rotate_coord(x, y, PA):
        x_rot = x * np.sin(np.radians(PA)) + y * np.cos(np.radians(PA))
        y_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
        return x_rot, y_rot

    @staticmethod
    def _deproject_coord(x, y, incl):
        return x, y / np.cos(np.radians(incl))

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
        self.Omega_beam_arcsec2 = np.pi / (4 * np.log(2)) * self.bmaj * self.bmin
        self.Omega_beam_str = self.Omega_beam_arcsec2 * arcsec_to_rad**2

        ### tuple of bmaj, bmin, and bpa
        # self.beam = (self.bmaj, self.bmin, self.bpa)

        # return self.beam # to make accesible from outside

    @property
    def ndim(self):
        return self.data.ndim

    @ndim.setter
    def ndim(self, ndim):
        self.ndim = ndim

    @property
    def shape(self):
        return self.data.shape

    @property
    def beam(self):
        return self.bmaj, self.bmin, self.bpa

    @property
    def restfreq(self):
        return self.nu0

    # @property
    # def nu0(self):
    #     return self.nu0

    @property
    def nxpix(self):
        return self.x.size

    @property
    def nypix(self):
        return self.y.size

    @property
    def dpix(self):
        return abs(np.diff(self.x).mean())

    @property
    def nchan(self):
        return self.v.size

    @nchan.setter
    def nchan(self, nchan):
        self.nchan = nchan



    def get_threshold_mask(self, threshold=3):
        self.SNR = self.data / self.rms
        self.threshold_mask = self.SNR >= threshold

    def get_mask(
        self,
        x0=0.0,
        y0=0.0,
        rmin=0.0,
        rmax=np.inf,
        thetamin=-180.0,
        thetamax=180.0,
        PA=0.0,
        incl=0.0,
        vmin=-np.inf,
        vmax=np.inf,
    ):
        # get projected coordinate
        r, t = self.get_disk_coord(x0=x0, y0=y0, PA=PA, incl=incl, frame="polar")

        # radial mask
        r_mask = np.logical_and(r >= rmin, r <= rmax)

        # azimuthal mask
        t_mask = np.logical_and(t >= thetamin, t <= thetamax)

        # channel mask
        if self.ndim > 2:
            # have 3D directinal masks
            r_mask = np.expand_dims(r_mask, axis=0)
            t_mask = np.expand_dims(t_mask, axis=0)

            # calculate velocity axis
            c_mask = np.logical_and(self.v >= vmin, self.v <= vmax)
            c_mask = np.expand_dims(c_mask, axis=(1, 2))

        # combine
        self.mask = np.logical_and(r_mask, t_mask)

        if self.ndim > 2:
            self.mask = np.logical_and(self.mask, c_mask)

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

        self.get_mask(**mask_kwargs)

        if edgenchan is not None:
            data = np.concatenate((self.data[:edgenchan], self.data[-edgenchan:]))
            mask = np.concatenate((self.mask[:edgenchan], self.mask[-edgenchan:]))
            self.rms = np.nanstd(data[mask])

        else:
            self.rms = np.nanstd(self.data[self.mask])

        return self.rms

    @staticmethod
    def estimate_rms_each_chan(data, mask):

        rms = np.array(np.nanstd(d[m]) for d, m in zip(data, mask))

        return rms

    def extract_peak_spectrum(self, peak_coord=None, frame="icrs", **mask_kwargs):

        self.get_mask(**mask_kwargs)

        if peak_coord is None:
            x0, y0 = self.get_phasecenter_coord()
        else:
            if isinstance(peak_coord, str):
                peak_coord = SkyCoord(peak_coord, frame=frame)
            x0, y0 = peak_coord.ra.arcsec, peak_coord.dec.arcsec

        x, y = self._get_directional_axis(relative=False)

        # set velocity range
        vrange = (mask_kwargs.get("vmin", -np.inf), mask_kwargs.get("vmax", np.inf))
        v = self.v[is_within(self.v, vrange)]
        data = self.data[is_within(self.v, vrange)] 
        mask = self.mask[is_within(self.v, vrange)]

        spec = np.squeeze([interp2d(y, x, d*m)(y0, x0) for d, m in zip(data, mask)])
        std = self.estimate_rms_each_chan(data, mask)

        return v, spec, std

    def extract_averaged_spectrum(self, user_mask=None, **mask_kwargs):
        self.get_mask(**mask_kwargs)

        if user_mask is not None:
            self.mask *= user_mask

        # set velocity range
        vrange = (mask_kwargs.get("vmin", -np.inf), mask_kwargs.get("vmax", np.inf))
        v = self.v[is_within(self.v, vrange)]
        data = self.data[is_within(self.v, vrange)] 
        mask = self.mask[is_within(self.v, vrange)]

        spec = np.squeeze(
            [np.nanmean(d[m]) for d, m in zip(data, mask)]
        )
        std = np.squeeze(
            [np.nanstd(d[m]) for d, m in zip(data, mask)]
        )

        return v, spec, std

    @staticmethod
    def get_spectroscopic_data_text(mol, line_data):
        text = "{:s} $\log_{{10}}A = {:.2f}$ $E_\mathrm{{u}} = {:.1f}$ K".format(
            # line_data["Species"],
            mol,
            line_data["logA [s^-1]"],
            line_data["E_u [K]"],
        )
        return text

    def plot_spectrum(
        self,
        ax=None,
        xval="velocity",
        yerr=False,
        baseline=None,
        vsys=None,
        plot_vsys=True,
        linedata_dict=None,
        linecolor=None,
        **kwargs
    ):

        if ax is None:
            fig, ax = plt.subplots()

        if "vel" in xval:
            x = self.v
        elif "freq" in xval:
            x = self.nu
        elif "chan" in xval:
            x = np.arange(self.v.size)
        else:
            raise ValueError(
                "{} for *xval* not supported. Should be 'velocity', 'frequency', or 'channel'.".format(
                    xval
                )
            )

        if yerr:
            ax.errorbar(**kwargs)
        else:
            ax.plot(x, self.avgspec, **kwargs)

        ax.set(xlim=(x.min(), x.max()))

        if baseline is not None:
            ax.axhline(y=baseline, color="grey", ls="dashed")
        if plot_vsys:
            if vsys is None:
                print("Warning: Provide *vsys* if you want to add vertical line for systemic velocity.")
            else:
                ax.axvline(x=vsys, color="grey", ls="dotted")

        if linedata_dict is not None:
            for mol in linedata_dict.keys():
                for linedata in linedata_dict[mol]:
                    vsys = 0.0 if vsys is None else vsys
                    if "vel" in xval:
                        x = (1 - float(linedata["nu0 [GHz]"])*1e9 / self.nu0) * ckms + vsys
                    else:
                        dnu = - self.nu0 * vsys / ckms
                        x = float(linedata["nu0 [GHz]"])*1e9 + dnu
                        if "chan" in xval:
                            x = (x - self.nu.min()) / np.diff(self.nu).mean()

                    ax.axvline(x=x, color=linecolor[mol], ls="dashed", lw=1.0)

                    # annotate line data info
                    desc = self.get_spectroscopic_data_text(mol, linedata)
                    ymin, _ = ax.get_ylim()
                    ax.text(x=x, y=ymin, s=desc, ha="left", va="bottom", rotation=90, color=linecolor[mol])

        try:
            return fig, ax
        except UnboundLocalError:
            return

    def spectrally_collapse(
        self, vrange=None, sigma_clip=None, rms=None, noiseedgenchan=3, mode="average"
    ):

        if self.ndim < 3:
            raise ValueError("The image is not 3D. Spectral collapse is not avilable.")

        if rms is None:
            self.estimate_rms(edgenchan=noiseedgenchan)
            rms = self.rms

        data = self.data

        # self.get_spectral_coord()
        v = self.v

        if vrange is not None:
            data = data[is_within(self.v, vrange), :, :]
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

    # def get_directional_coord(self, center_coord=None, in_arcsec=True):
    #     """Generate a (RA\cos(Dec), Dec) coordinate (1D each) in arcsec. Assume the unit for coordinates in the header is deg.

    #     Args:
    #         header (dict): FITS header.
    #         center_coord (tuple or astropy.coordinates.SkyCoord object, optinal): Two component tuple of (RA, Dec) in arcsec or the SkyCoord object for the center coordinate. Defaults to (0.0, 0.0)

    #     Returns:
    #         tuple: Coordinates
    #     """

    #     x0 = self.x0
    #     y0 = self.y0
    #     dx = self.dx
    #     dy = self.dy

    #     if in_arcsec:
    #         assert self.ux == self.uy == "deg"
    #         x0 *= deg_to_arcsec
    #         y0 *= deg_to_arcsec
    #         dx *= deg_to_arcsec
    #         dy *= deg_to_arcsec
    #     if center_coord is None:
    #         offset_x, offset_y = 0, 0
    #     else:
    #         if isinstance(center_coord, tuple):
    #             center_x, center_y = center_coord
    #         elif isinstance(center_coord, SkyCoord):
    #             center_x = center_coord.ra.arcsec
    #             center_y = center_coord.dec.arcsec
    #         elif isinstance(center_coord, str):
    #             center_coord = SkyCoord(center_coord, frame="icrs")
    #             center_x = center_coord.ra.arcsec
    #             center_y = center_coord.dec.arcsec
    #         offset_x = center_x - x0  # offset along x from phsecenter in arcsec
    #         offset_y = center_y - y0  # offset along y from phsecenter in arcsec

    #     self.x = dx * (np.arange(self.nx) - (self.header["CRPIX1"] - 1)) - offset_x
    #     self.y = dy * (np.arange(self.ny) - (self.header["CRPIX2"] - 1)) - offset_y

    # def get_projected_coord(self, PA=90.0, incl=0.0, center_coord=None, in_arcsec=True):
    #     self.get_directional_coord(center_coord=center_coord, in_arcsec=in_arcsec)

    #     # meshgrid to be in 2D
    #     xx, yy = np.meshgrid(self.x, self.y)

    #     # project to the disk plane; assume geometrically thin disk
    #     incl = np.radians(incl)
    #     PA = np.radians(PA)

    #     self.x_proj = xx * np.sin(PA) + yy * np.cos(PA)
    #     self.y_proj = (-xx * np.cos(PA) + yy * np.sin(PA)) / np.cos(
    #         incl
    #     )  # follow the formulation in Yen et al. 2016

    #     # polar coordinate
    #     self.r = np.sqrt(self.x_proj**2 + self.y_proj**2)  # in arcsec
    #     self.theta = np.degrees(
    #         np.arctan2(self.y_proj, self.x_proj)
    #     )  # in degree, [-180, 180]

    # def get_spectral_coord(self):
    #     if self.ndim < 3:
    #         raise KeyError("Spectral axis not found.")

    #     # assume in frequency
    #     self.nu = self.dz * (np.arange(self.nz) - (self.header["CRPIX3"] - 1)) + self.z0

    #     assert self.header["VELREF"] == 257  # in radio convention
    #     self.v = ckms * (1 - self.nu / self.restfreq)

    # def cutout(self, xlim=None, ylim=None, vlim=None):
    #     if xlim is not None:
    #         self.data = self.data[..., is_within(self.x, xlim)]
    #         try:
    #             self.x_proj = self.x_proj[:, is_within(self.x, xlim)]
    #             self.y_proj = self.y_proj[:, is_within(self.x, xlim)]
    #             self.r = self.r[:, is_within(self.x, xlim)]
    #             self.theta = self.theta[:, is_within(self.x, xlim)]
    #         except AttributeError:
    #             pass
    #         self.x = self.x[is_within(self.x, xlim)]

    #     if ylim is not None:
    #         self.data = self.data[..., is_within(self.y, ylim), :]
    #         try:
    #             self.x_proj = self.x_proj[is_within(self.y, ylim), :]
    #             self.y_proj = self.y_proj[is_within(self.y, ylim), :]
    #             self.r = self.r[is_within(self.y, ylim), :]
    #             self.theta = self.theta[is_within(self.y, ylim), :]
    #         except AttributeError:
    #             pass
    #         self.y = self.y[is_within(self.y, ylim)]

    #     if vlim is not None:
    #         self.data = self.data[is_within(self.v, vlim)]
    #         self.nu = self.nu[is_within(self.v, vlim)]
    #         self.v = self.v[is_within(self.v, vlim)]

    # def downsample(self, N):
    #     # adopted from eddy code by rich teague
    #     N = int(np.ceil(self.bmaj / (abs(self.dx)*deg_to_arcsec))) if N == 'beam' else N
    #     N0x, N0y = int(N / 2), int(N / 2)
    #     if N > 1:
    #         self.x = self.x[N0x::N]
    #         self.y = self.y[N0y::N]
    #         self.data = self.data[..., N0y::N, N0x::N]
    #         try:
    #             self.x_proj = self.x_proj[N0y::N, N0x::N]
    #             self.y_proj = self.y_proj[N0y::N, N0x::N]
    #             self.r = self.r[N0y::N, N0x::N]
    #             self.theta = self.theta[N0y::N, N0x::N]
    #             # self.mask = self.mask[N0y::N, N0x::N]
    #         except AttributeError:
    #             pass
    #         # self.error = self.error[N0y::N, N0x::N]
    #         # self.mask = self.mask[N0y::N, N0x::N]


class PVFitsImage:
    def __init__(self, fitsname, squeeze=True, xlim=None, vlim=None, downsample=False):

        self.fitsname = fitsname

        # header
        self.header = fits.getheader(fitsname)

        # data
        self.data = fits.getdata(fitsname)
        if squeeze:
            self.data = np.squeeze(self.data)

        self._get_PVFITS_properties()

        self._cutout(xlim=xlim, vlim=vlim)

        if downsample:
            if not isinstance(downsample, tuple):
                downsample = (downsample, 1)
            self.downsample_cube(*downsample)

    def _get_PVFITS_properties(self):

        # data unit
        self.data_unit = self.header["bunit"]

        # directional axis
        self.posax = self._get_positional_axis()

        # spectral axis
        self.nu0 = self._get_restfreq()
        self.nu, self.v = self._get_spectral_axis()

        # beam
        try:
            self._get_beam_info()
        except KeyError:
            print("Warning; No beam information found.")
            pass

    def _get_axis(self, n):
        npix = self.header["naxis{:d}".format(n)]
        delta = self.header["cdelt{:d}".format(n)]
        rpix = self.header["crpix{:d}".format(n)]
        rval = self.header["crval{:d}".format(n)]

        axis = delta * (np.arange(npix) - rpix + 1) + rval

        return axis

    def _get_positional_axis(self):
        posax = self._get_axis(n=1)
        return posax

    def _get_spectral_axis(self):
        if "Hz" in self.header["cunit2"]:
            nu = self._get_axis(n=2)
            v = (self.nu0 - nu) * ckms / self.nu0
        else:
            v = self._get_axis(n=2)
            vunit = self._get_axis_unit_in_header(n=2)
            if not "k" in vunit:  # the case for in m/s
                v *= 1e3  # in km/s
            nu = (1 - v / ckms) * self.nu0
        return nu, v

    def _get_axis_unit_in_header(self, n):
        return self.header["cunit{:d}".format(n)]

    def _get_restfreq(self):
        try:
            return self.header["restfreq"]
        except KeyError:
            try:
                return self.header["restfrq"]
            except KeyError:
                if self.ndim > 2:
                    print("Warning: no rest frequency is found in header.")
                return np.nan

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
        self.Omega_beam_arcsec2 = np.pi / (4 * np.log(2)) * self.bmaj * self.bmin
        self.Omega_beam_str = self.Omega_beam_arcsec2 * arcsec_to_rad**2

    def _cutout(self, xlim=None, vlim=None):
        if xlim is not None:
            self.data = self.data[:, is_within(self.posax, xlim)]
            self.posax = self.posax[is_within(self.posax, xlim)]
        if vlim is not None:
            self.data = self.data[is_within(self.v, vlim), :]
            self.v = self.v[is_within(self.v, vlim)]

    def _downsample_spatial(self, N):
        # adopted from eddy code by rich teague
        N = int(np.ceil(self.bmaj / abs(self.dpix))) if N == "beam" else N
        if N > 1:
            self.posax = self.x[::N]
            self.data = self.data[:, ::N]

    def _downsample_spectral(self, N):
        if self.ndim < 3:
            raise ValueError("Spectral downsample is not available for 2D image.")
        if N > 1:
            self.v = self.v[::N]
            self.nu = self.nu[::N]
            self.data = self.data[::N, :]

    def downsample_cube(self, Nxy, Nv):
        self._downsample_spatial(Nxy)
        self._downsample_spectral(Nv)

    def restore_original_cube(self, squeeze=True):
        self.data = fits.getdata(self.fitsname)
        if squeeze:
            self.data = np.squeeze(self.data)

        self._get_PVFITS_properties()

    @property
    def ndim(self):
        return self.data.ndim

    @ndim.setter
    def ndim(self, ndim):
        self.ndim = ndim

    @property
    def shape(self):
        return self.data.shape

    @property
    def beam(self):
        return self.bmaj, self.bmin, self.bpa

    @property
    def restfreq(self):
        return self.nu0

    @property
    def npix(self):
        return self.x.size

    @property
    def dpix(self):
        return abs(np.diff(self.x).mean())

    @property
    def nchan(self):
        return self.v.size

    @property
    def dchan(self):
        return abs(np.diff(self.v)).mean()


# class PVFitsImage(FitsImage):
#     def __init__(self, fitsname):
#         super().__init__(fitsname)
#         self._get_PVaxes_info()

#     def _get_PVaxes_info(self):
#         for i in range(self.ndim):
#             if "offset" in self.header["CTYPE{:d}".format(i + 1)].lower():
#                 # read off position axes info
#                 self.np = self.header["NAXIS{:d}".format(i + 1)]  # number of pixels
#                 self.dp = self.header["CDELT{:d}".format(i + 1)]  # increment
#                 self.p0 = self.header["CRVAL{:d}".format(i + 1)]  # reference value
#                 self.rp = self.header["CRPIX{:d}".format(i + 1)]  # reference pixel
#                 self.up = self.header["CUNIT{:d}".format(i + 1)]  # unit

#             if "freq" in self.header["CTYPE{:d}".format(i + 1)].lower():
#                 # read off velocity axis info in the case header is in frequency
#                 self.nv = self.header["NAXIS{:d}".format(i + 1)]  # number of pixels
#                 self.dv = (
#                     self.header["CDELT{:d}".format(i + 1)] * ckms / self.restfreq
#                 )  # increment
#                 self.v0 = ckms * (
#                     1.0 - self.header["CRVAL{:d}".format(i + 1)] / self.restfreq
#                 )  # reference value
#                 self.rv = self.header["CRPIX{:d}".format(i + 1)]  # reference pixel
#                 self.uv = "km/s"  # unit

#             elif "velocity" in self.header["CTYPE{:d}".format(i + 1)].lower():
#                 # read off velocityaxis info in the case header is in velocity
#                 self.nv = self.header["NAXIS{:d}".format(i + 1)]  # number of pixels
#                 self.dv = self.header["CDELT{:d}".format(i + 1)]  # increment
#                 self.v0 = self.header["CRVAL{:d}".format(i + 1)]  # reference value
#                 self.rv = self.header["CRPIX{:d}".format(i + 1)]  # reference pixel
#                 self.uv = self.header["CUNIT{:d}".format(i + 1)]  # unit

#             else:
#                 continue

#     def get_coord(self):
#         # fetch all coordinates
#         self.get_position_coord()
#         self.get_velocity_coord()

#     def get_position_coord(self):
#         self.p = self.dp * (np.arange(self.np) - (self.rp - 1)) + self.p0
#         # unit conversion
#         if self.up == "deg":
#             self.p *= deg_to_arcsec
#         if self.up == "rad":
#             self.p *= rad_to_arcsec

#     def get_velocity_coord(self):
#         self.v = self.dv * (np.arange(self.nv) - (self.rv - 1)) + self.v0
