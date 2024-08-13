import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp2d, griddata
from astropy.convolution import Gaussian2DKernel, convolve_fft
import numpy as np
import astropy.constants as ac
import astropy.units as u
import matplotlib.pyplot as plt
from .utils import is_within
import casatools, casatasks
from qdisk.utils import (
    remove_casalogfile,
    jypb_to_K_RJ,
    jypb_to_K,
    bettermoments_collapse_wrapper,
    process_chunked_array,
    moment_method,
)
import bettermoments as bm
from scipy.special import ellipe

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


class BoundingBox:
    def __init__(self, x, y, array, pad=0.0, collapse_to_2D=False):
        self.x_grid, self.y_grid = np.meshgrid(x, y)
        array = np.sum(array, axis=0)[None, :, :] if collapse_to_2D else array
        self.bbox = np.empty_like(array)
        nchan = array.shape[0]
        self.xmin = np.empty(nchan)
        self.xmax = np.empty(nchan)
        self.ymin = np.empty(nchan)
        self.ymax = np.empty(nchan)
        self.xsize = np.empty(nchan)
        self.ysize = np.empty(nchan)

        for i, a in enumerate(array):
            self.xmin[i] = np.nanmin(self.x_grid[a != 0.0]) - pad
            self.xmax[i] = np.nanmax(self.x_grid[a != 0.0]) + pad
            self.ymin[i] = np.nanmin(self.y_grid[a != 0.0]) - pad
            self.ymax[i] = np.nanmax(self.y_grid[a != 0.0]) + pad
            self.bbox[i, :, :] = (
                (self.x_grid >= self.xmin[i])
                * (self.x_grid <= self.xmax[i])
                * (self.y_grid >= self.ymin[i])
                * (self.y_grid <= self.ymax[i])
            )
            self.xsize[i] = self.xmax[i] - self.xmin[i]
            self.ysize[i] = self.ymax[i] - self.ymin[i]

        self.xmin = np.squeeze(self.xmin)
        self.xmax = np.squeeze(self.xmax)
        self.ymin = np.squeeze(self.ymin)
        self.ymax = np.squeeze(self.ymax)
        self.bbox = np.squeeze(self.bbox)
        self.xsize = np.squeeze(self.xsize)
        self.ysize = np.squeeze(self.ysize)


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
        FoV=None,
        xlim=None,
        ylim=None,
        vlim=None,
        nu0=None,
        downsample=False,
        skipdata=False,
    ):
        self.fitsname = fitsname

        # header
        self.header = fits.getheader(fitsname)

        # read header
        # self.ndim = self.header["NAXIS"]
        # self.data_unit = self.header["BUNIT"]

        # axis etc.
        self.rel_dir_ax = rel_dir_ax

        self._get_FITS_properties(rel_dir_ax=self.rel_dir_ax, nu0=nu0)

        if not skipdata:
            # data
            self.data = fits.getdata(fitsname)
            if squeeze:
                self.data = np.squeeze(self.data)

            if FoV is not None:
                self.xlim = (-FoV * 0.5, FoV * 0.5)
                self.ylim = (-FoV * 0.5, FoV * 0.5)
            else:
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

        axis = delta * (np.arange(npix) - rpix + 1.0) + rval

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
                v *= 1e-3  # in km/s
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
                print("Warning: no rest frequency is found in header.")
                return np.nan

    def _get_FITS_properties(self, rel_dir_ax=True, nu0=None):
        # data unit
        self.data_unit = self.header["bunit"]

        # directional axis
        self.x, self.y = self._get_directional_axis(relative=rel_dir_ax)
        self.x -= 0.5 * self.dpix
        self.y -= 0.5 * self.dpix

        # spectral axis
        self.nu0 = self._get_restfreq() if nu0 is None else nu0
        try:
            self.nu, self.v = self._get_spectral_axis()
        except:
            pass

        # beam
        try:
            self._get_beam_info()
        except KeyError:
            print("Warning: No beam information found in the header.")
            pass

    def convert_unit(self, unit="K", nu=None, RJ_approx=False):
        ### assume the original data unit is Jy / beam or mJy / beam
        if not np.isnan(self.restfreq):
            nu = self.restfreq
        elif nu is None:
            raise ValueError(
                "Rest frequency not found. Please provide it via *nu* argument."
            )

        data = self.data
        if "mJy" in self.data_unit:
            data *= 1e-3  # in Jy /beam

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
            x0 = self.header["crval1"] * deg_to_arcsec - 0.5 * self.dpix
            y0 = self.header["crval2"] * deg_to_arcsec - 0.5 * self.dpix
        else:
            x0, y0 = self.get_phasecenter_coord()
        c_ref = SkyCoord(ra=x0, dec=y0, unit=u.arcsec, frame="icrs")
        dx, dy = c_ref.spherical_offsets_to(c)
        self.x -= dx.to(u.arcsec).value
        self.y -= dy.to(u.arcsec).value

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

    @property
    def dchan(self):
        return np.abs(np.diff(self.v).mean())

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
        abs_theta=False,
        exclude_theta=False,
        PA=0.0,
        incl=0.0,
        vmin=-np.inf,
        vmax=np.inf,
        user_mask=None,
        combine="and",
        convolve=False,
        tolerance=0.01,
    ):
        # get projected coordinate
        r, t = self.get_disk_coord(x0=x0, y0=y0, PA=PA, incl=incl, frame="polar")

        # radial mask
        r_mask = np.logical_and(r >= rmin, r <= rmax)

        # azimuthal mask
        t_mask = np.logical_and(t >= np.radians(thetamin), t <= np.radians(thetamax))
        if abs_theta:
            t_mask = np.logical_or(
                t_mask,
                np.logical_and(t >= -np.radians(thetamax), t <= -np.radians(thetamin)),
            )
        if exclude_theta:
            t_mask = np.logical_not(t_mask)

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

        if user_mask is not None:
            combine = getattr(np, "logical_" + combine)
            if user_mask.ndim <= 2 and self.ndim > 2:
                user_mask = np.expand_dims(user_mask, axis=0)
            self.mask = combine(self.mask, user_mask)

        if convolve:
            beam_kernel = Gaussian2DKernel(
                x_stddev=self.bmaj / self.dpix,
                y_stddev=self.bmin / self.dpix,
                theta=np.radians(90.0 - self.bpa),
            )
            self.mask = convolve_fft(self.mask, beam_kernel)
            self.mask = self.mask / np.nanmax(self.mask) > tolerance

        return self.mask

    def get_mask_bounding_box(self, pad=0.0, collapse_to_2D=False):
        self.mask_bbox = BoundingBox(
            self.x, self.y, self.mask, pad=pad, collapse_to_2D=collapse_to_2D
        )
        return self.mask_bbox

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
            self.rms = np.nanstd(data[mask & (data != 0.0)])

        else:
            self.rms = np.nanstd(self.data[self.mask & (self.data != 0.0)])

        return self.rms

    @staticmethod
    def estimate_rms_each_chan(data, mask):
        rms = np.array([np.nanstd(d[m.astype(bool) & (d != 0.0)]) for d, m in zip(data, mask)])

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

        spec = np.squeeze([interp2d(y, x, d * m)(y0, x0) for d, m in zip(data, mask)])
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

        spec = np.squeeze([np.nanmean(d[m]) for d, m in zip(data, mask)])
        std = np.squeeze([np.nanstd(d[m]) for d, m in zip(data, mask)])

        return v, spec, std

    def extract_integrated_spectrum(self, rms=None, **mask_kwargs):
        mask = self.get_mask(**mask_kwargs)

        if self.ndim <= 2:
            data = np.expand_dims(self.data, axis=0)
            mask = np.expand_dims(mask, axis=0)
        else:
            data = self.data.copy()

        flux_spectrum = []
        flux_spectrum_error = []
        if (rms is None) and (not hasattr(self, "rms")):
            print(
                "Warning: Flux uncertainty will not be calculated due to the lack of rms value. If you want it, provide the rms value or estimate by estimate_rms function."
            )

        for d, m in zip(data, mask):
            tointeg = d.flatten()[m.flatten()]
            flux = np.sum(tointeg / self.Omega_beam_arcsec2 * self.dpix**2)
            flux_spectrum.append(flux)
            if rms is not None:
                flux_error = rms * np.sqrt(
                    tointeg.size * self.dpix**2 / self.Omega_beam_arcsec2
                )  # / self.Omega_beam_arcsec2 * self.dpix**2
            elif hasattr(self, "rms"):
                flux_error = self.rms * np.sqrt(
                    tointeg.size * self.dpix**2 / self.Omega_beam_arcsec2
                )
            else:
                flux_error = 0.0
            flux_spectrum_error.append(flux_error)

        return self.v, np.array(flux_spectrum), np.array(flux_spectrum_error)

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
                print(
                    "Warning: Provide *vsys* if you want to add vertical line for systemic velocity."
                )
            else:
                ax.axvline(x=vsys, color="grey", ls="dotted")

        if linedata_dict is not None:
            for mol in linedata_dict.keys():
                for linedata in linedata_dict[mol]:
                    vsys = 0.0 if vsys is None else vsys
                    if "vel" in xval:
                        x = (
                            1 - float(linedata["nu0 [GHz]"]) * 1e9 / self.nu0
                        ) * ckms + vsys
                    else:
                        dnu = -self.nu0 * vsys / ckms
                        x = float(linedata["nu0 [GHz]"]) * 1e9 + dnu
                        if "chan" in xval:
                            x = (x - self.nu.min()) / np.diff(self.nu).mean()

                    ax.axvline(x=x, color=linecolor[mol], ls="dashed", lw=1.0)

                    # annotate line data info
                    desc = self.get_spectroscopic_data_text(mol, linedata)
                    ymin, _ = ax.get_ylim()
                    ax.text(
                        x=x,
                        y=ymin,
                        s=desc,
                        ha="left",
                        va="bottom",
                        rotation=90,
                        color=linecolor[mol],
                    )

        try:
            return fig, ax
        except UnboundLocalError:
            return

    def extract_flux(
        self, rms=None, verbose=True, velocity_resolution=1.0, **mask_kwargs
    ):
        """Measure the (velocity-integrated) flux density in a specified region.

        Parameters
        ----------
        rms : float, optional
            RMS noise of the image, by default None. If None, it will be internally calculated over
            the outside of the mask specified by the mask_kwargs paramaters.
        verbose : bool, optional
            Whether the message about the flux denisty will be shown in the terminal, by default True
        velocity_resolution : float, optional
            velocity resolution in unit of the channel width which is used to correct for the spectral correlation, by default 1.0.

        Returns
        -------
        tuple of two floats
            (velocity-integrated) flux density and its uncertainty
        """
        _, flux_spectrum, flux_spectrum_error = self.extract_integrated_spectrum(
            rms=rms, **mask_kwargs
        )

        if len(flux_spectrum) == 1:
            flux = flux_spectrum[0]
            flux_error = flux_spectrum_error[0]
            if verbose:
                print("Extracted flux density: {:.3e} Jy".format(flux))
                print(
                    "Extracted flux density uncertainty: {:.3e} Jy".format(flux_error)
                )
        else:
            flux = self.dchan * np.sum(flux_spectrum)
            flux_error = (
                self.dchan
                * np.sqrt(np.sum(flux_spectrum_error**2))
                * np.sqrt(velocity_resolution)
            )
            if verbose:
                print("Extracted integrated flux density: {:.3e} Jy km/s".format(flux))
                print(
                    "Extracted integrated flux density uncertainty: {:.3e} Jy km/s".format(
                        flux_error
                    )
                )

        return flux, flux_error

    def get_cumulative_flux(
        self, rms=None, PA=0.0, incl=0.0, rmin=0.0, dr="beam", rmax=None, criteria="val"
    ):
        r, _ = self.get_disk_coord(PA=PA, incl=incl)
        dr = self.bmaj if dr == "beam" else self.dpix if dr == "pix" else dr
        rmax = np.nanmax(r) if rmax is None else rmax
        mask_radii = np.arange(rmin, rmax + dr, dr)

        f0 = 0.0
        df0 = 0.0
        radii = []
        cum_f = []
        cum_df = []

        print("Applying curve-of-growth method...")
        for r in mask_radii:
            f, df = self.get_flux(rms=rms, PA=PA, incl=incl, rmin=0.0, rmax=r)
            radii.append(r)
            cum_f.append(f)
            cum_df.append(df)

            if criteria == "val":
                condition = f < f0
            elif "sigma" in criteria:
                condition = f < f0 + float(criteria.replace("sigma", "")) * df0

            if condition:
                break
            else:
                f0 = f
                df0 = df

        return radii, cum_f, cum_df

    # def get_cumurative_flux(
    #     self,
    #     PA=0.0,
    #     incl=45.0,
    #     rbins=None,
    #     rmin=0.0,
    #     rmax=None,
    #     wedge_angle=None,
    #     assume_correlated=True,
    #     **mask_kwargs
    # ):

    #     r, I, dI = self.radial_profile(
    #         PA=PA,
    #         incl=incl,
    #         rbins=rbins,
    #         rmin=rmin,
    #         rmax=rmax,
    #         wedge_angle=wedge_angle,
    #         assume_correlated=assume_correlated,
    #         **mask_kwargs
    #     )

    #     # covert to the flux
    #     f = I *

    #     return

    # def curve_of_growth(self, PA=0.0, incl=0.0, start=0.0, stop=None, dr="beam", plot=False, ax=None):
    #     if dr == "beam":
    #         dr = self.bmaj
    #     elif dr == "pix":
    #         dr = self.dpix

    #     if plot and ax is None:
    #         fig, ax = plt.subplots()

    #     stop = np.nanmax(self.x) if stop is None else stop
    #     r_array = np.arange(start, stop+dr, dr)
    #     f_prev = 0.0

    #     for r in r_array:
    #         self.get_mask(PA=PA, incl=incl, rmax=r)
    #         f = np.sum(self.data * self.mask) / self.Omega_beam_arcsec2 * self.dpix**2

    #         if f < f_prev:
    #             break
    #         else:
    #             f_prev = f
    #             if plot:
    #                 ax.scatter(r, f, color="tab:blue")

    #     if plot:
    #         return r, f, ax
    #     else:
    #         return r, f, None

    def _extract_cut(self, x, y, x_ip, y_ip, pad=10):
        # mask out non-relevant image
        pad = self.dpix * pad  # 10 pixel padding
        mask = is_within(x, (x_ip.min() - pad, x_ip.max() + pad)) & is_within(
            y, (y_ip.min() - pad, y_ip.max() + pad)
        )

        x, y = x[mask].flatten(), y[mask].flatten()

        cut = griddata(
            np.array([y, x]).T, self.data[mask].flatten(), (y_ip, x_ip), method="cubic"
        )

        return cut

    def cut(
        self,
        PA=0.0,
        incl=45.0,
        rmax=None,
        range=None,
        axis="major",
        offset=0.0,
        side_average=True,
    ):
        major, minor = self.get_disk_coord(PA=PA, incl=incl, frame="cartesian")

        # get interpolate axis
        if range is not None:
            x_ip = np.arange(*range, self.dpix)  # x for the axis along the profile cut
        else:
            rmax = np.nanmax(minor) if rmax is None else rmax
            x_ip_oneside = np.arange(
                self.dpix * 0.5, rmax, self.dpix
            )  # x for the axis along the profile cut
            x_ip = np.concatenate((-x_ip_oneside[::-1], x_ip_oneside))

        y_ip = np.full_like(x_ip, offset)

        # set the original (x,y) axis
        x, y = (major, minor) if axis == "major" else (minor, major)

        # interpolation (cubic)
        r = x_ip
        cut = self._extract_cut(x, y, x_ip, y_ip)
        cut_err = np.full_like(cut, self.rms)

        if side_average:
            r = x_ip_oneside
            cut = np.mean([cut[x_ip > 0], cut[x_ip < 0][::-1]], axis=0)
            cut_err = np.full_like(cut, self.rms) / np.sqrt(2)
            x_ip = x_ip_oneside

        return r, cut, cut_err

    def pvcut(
        self,
        PA=0.0,
        incl=0.0,
        xrange=None,
        vrange=None,
        axis="major",
        offset=0.0,
        width=None,
        save=False,
        savefilename=None,
    ):
        if self.ndim < 3:
            raise AssertionError(
                "Dimension of the data is not in 3D. pvcut is not possible."
            )

        major, minor = self.get_disk_coord(PA=PA, incl=incl, frame="cartesian")

        x, y = (major, minor) if axis == "major" else (minor, major)

        # interpolation axis
        xrange = (x.min(), x.max()) if xrange is None else xrange
        x_interp = np.arange(*xrange, self.dpix)
        x_interp = np.append(x_interp, x_interp.max() + self.dpix)

        # store the positional axis and data and velcoity axis
        posax = x_interp.copy()
        velax = self.v.copy()
        data = self.data.copy()

        # manage width parameter
        if width is None:
            width = 0.0
            y_interp = np.full_like(
                x_interp, offset
            )  # trace y = offset on the projected coord
        else:
            y_interp = np.arange(offset - width / 2, offset + width / 2, self.dpix)
            y_interp = np.append(y_interp, y_interp.max() + self.dpix)

            # for interpolation
            x_interp, y_interp = np.meshgrid(x_interp, y_interp)
            x_interp = x_interp.flatten()
            y_interp = y_interp.flatten()

        # masking to reduce the calcullation time
        pad = self.dpix * 10  # 10 pixel padding
        mask = is_within(x, (xrange[0] - pad, xrange[1] + pad)) & is_within(
            y, (-width / 2 - pad, width / 2 + pad)
        )
        x = x[mask].flatten()
        y = y[mask].flatten()

        # velocity mask
        if vrange is not None:
            v_mask = is_within(velax, vrange)
            data = data[v_mask, :, :]  # exclude non-relevant channels
            velax = velax[v_mask]

        print("Calculating PV diagram...")
        # interpolate over the cube
        pvdiagram = np.array(
            [
                griddata(
                    np.array([y, x]).T,
                    im[mask].flatten(),
                    (y_interp, x_interp),
                    method="cubic",
                )
                for im in data
            ]
        )

        # average along y axis if width is not None
        if width is not None:
            pvdiagram = np.nanmean(
                pvdiagram.reshape(velax.size, -1, posax.size), axis=1
            )

        print("Done.")

        if save:
            # modify the header for PV diagram; will drop any degenerate axes
            header = self.header.copy()
            header["NAXIS"] = 2
            header["NAXIS1"] = posax.size
            header["NAXIS2"] = velax.size
            # offset axis
            header["CTYPE1"] = "OFFSET"
            header["CRPIX1"] = np.ceil(posax.size / 2).astype(np.float32)
            header["CDELT1"] = self.dpix
            header["CRVAL1"] = posax[int(header["CRPIX1"]) - 1].astype(np.float32)
            header["CUNIT1"] = "arcsec"
            # frequency axis
            frqax = self.nu
            if vrange is not None:
                frqax = frqax[v_mask]
            header["CTYPE2"] = "FREQ"
            header["CRPIX2"] = np.ceil(frqax.size / 2).astype(np.float32)
            header["CDELT2"] = np.diff(frqax)[0].astype(np.float32)
            header["CRVAL2"] = frqax[int(header["CRPIX2"]) - 1].astype(np.float32)
            header["CUNIT2"] = "Hz"
            # remove No. 3 axis
            for key in list(header.keys()):
                if "3" in key:
                    del header[key]
            # comment
            header["COMMENT"] = "PV diagram generated by python script"

            if savefilename is None:
                savefilename = self.fitsname.replace(".fits", ".pv.fits")

            print("Saving PV diagram into {:s}...".format(savefilename))
            fits.writeto(
                savefilename,
                pvdiagram.astype(float),
                header,
                overwrite=True,
                output_verify="silentfix",
            )

        return posax, velax, pvdiagram

    def moment_map(
        self,
        moment="0",
        rms=None,
        noisechannel=3,
        mask=None,
        channel=None,
        vel_extent=None,
        threshold=None,
        save=True,
        savefilename=None,
        nchunks=None,
        verbose=False,
    ):
        if rms is None and not hasattr(self, "rms"):
            if verbose:
                print("Estimating rms...")
            self.estimate_rms(edgenchan=noisechannel)
            if verbose:
                print("rms: {} mJy/beam".format(rms * 1e3))

        rms = self.rms.copy() if rms is None else rms
        data = self.data.copy()
        velax = self.v.copy()  # velocity axis for bettermoments in m/s
        # if np.all(np.diff(velax) < 0):
        #     data = data[::-1, :, :]
        #     velax = velax[::-1]

        # user mask
        mask = np.ones(data.shape) if mask is None else mask

        # threshold mask
        if verbose:
            print("Generating threshold mask...")
        if threshold is not None:
            tmask = bm.get_threshold_mask(data=data, clip=threshold)
        else:
            tmask = np.ones(data.shape)

        # channel mask
        if channel is None and vel_extent is None:
            if verbose:
                print("Generating channel mask...")
            cmask = np.ones(data.shape)
        elif channel is not None:
            if verbose:
                print("Generating channel mask based on specified channels...")
            cmask = np.zeros(data.shape)
            for cr in channel.split(";"):
                firstchannel, lastchannel = [int(c) for c in cr.split("~")]
                cmask += bm.get_channel_mask(
                    data=data, firstchannel=firstchannel, lastchannel=lastchannel
                )
            cmask = np.where(cmask != 0.0, 1.0, 0.0)  # manage possible overlaps
        else:
            if verbose:
                print("Generating channel mask based on specified velocity range...")
            if isinstance(vel_extent, list):
                cmask = np.zeros(data.shape)
                for extent in vel_extent:
                    firstchannel, lastchannel = [
                        np.argmin(np.abs(velax - v)) for v in extent
                    ]
                    if firstchannel > lastchannel:
                        tmp = lastchannel
                        lastchannel = firstchannel
                        firstchannel = tmp
                    cmask += bm.get_channel_mask(
                        data=data, firstchannel=firstchannel, lastchannel=lastchannel
                    )
                cmask = np.where(cmask != 0.0, 1.0, 0.0)
            else:
                firstchannel, lastchannel = [
                    np.argmin(np.abs(velax - v)) for v in vel_extent
                ]
                if firstchannel > lastchannel:
                    tmp = lastchannel
                    lastchannel = firstchannel
                    firstchannel = tmp
                cmask = bm.get_channel_mask(
                    data=data, firstchannel=firstchannel, lastchannel=lastchannel
                )

        # mask combination
        if verbose:
            print("Combining the masks...")
        mask = bm.get_combined_mask(
            user_mask=mask, threshold_mask=tmask, channel_mask=cmask, combine="and"
        )

        # masked data
        data *= mask
        # if np.all(np.diff(velax) < 0):
        #     data = data[::-1, :, :]
        #     velax = velax[::-1]

        # moment calc by bettermoments; moment 1 may take time
        if verbose:
            print("Calculating moment {}...".format(moment))
        if nchunks is not None:
            if verbose:
                print("Going to compute with {} chunks...".format(nchunks))
            # note that bettermoment uses velocity unit of m/s
            moments = process_chunked_array(
                bettermoments_collapse_wrapper,
                data,
                moment=moment,
                velax=velax,
                rms=rms,
                nchunks=nchunks,
                axis=0,
            )
        else:
            # note that bettermoment uses velocity unit of m/s
            moments = bettermoments_collapse_wrapper(
                data=data, moment=moment, velax=velax, rms=rms
            )

        # workaround to force the unmasked region to be nan
        for m in moments:
            m[np.all(mask == 0.0, axis=0)] = np.nan

        if save:
            bm.save_to_FITS(
                moments=moments,
                method=moment_method[moment],
                path=self.fitsname,
                outname=savefilename,
            )
            saved_to = (
                savefilename.replace(".fits", "_*.fits")
                if savefilename is not None
                else self.fitsname.replace(".fits", "_*.fits")
            )
            # use velocity unit of km/s instead of m/s which is the default of bettermoments; to be compatible with several CASA tasks
            outputs = bm.collapse_method_products(method=moment_method[moment]).split(
                ", "
            )
            for ext in outputs:
                filename = saved_to.replace("*", ext)
                bunit = fits.getheader(filename)["BUNIT"]
                if " m/s" in bunit:
                    fits.setval(
                        filename=filename,
                        keyword="BUNIT",
                        value=bunit.replace("m/s", "km/s"),
                    )
            if verbose:
                print("Saved into {}.".format(saved_to))

        return moments

    def radial_sampling(self, PA=0.0, incl=45.0, rbins=None, rmin=0.0, rmax=None):
        r, _ = self.get_disk_coord(PA=PA, incl=incl)

        if rbins is None:
            rbin_width = self.bmaj / 4.0  # 1/4 of bmaj in arcsec
            if rmax is None:
                rmax = np.nanmax(r)
            rbins = np.arange(rmin, rmax, rbin_width)

        rvals = np.average([rbins[1:], rbins[:-1]], axis=0)

        return r, rvals, rbins

    def calc_nbeams(self, incl, r, rbins, rvals, npix):
        if (
            np.diff(rvals).mean() > (self.bmaj + self.bmin) * 0.5
        ):  # the radial bin width is larger than a beam width
            nbeams = npix * self.dpix**2 / self.Omega_beam_arcsec2
        else:  # consider the length of arc
            r = r.flatten()
            npix_full_wedge = np.array([r[(r >= rbins[idx]) & (r <= rbins[idx+1])].size for idx in range(rvals.size)])

            e = np.sin(np.radians(incl))
            nbeams = (
                4 * rvals[:, None] * ellipe(e) * npix / npix_full_wedge[:, None] / self.bmaj
            )

        return nbeams

    def radial_profile(
        self,
        PA=0.0,
        incl=45.0,
        rbins=None,
        rmin=0.0,
        rmax=None,
        wedge_angle=None,
        assume_correlated=True,
        save=False,
        savefilename=None,
        savefileheader="",
        **mask_kwargs
    ):
        r, rvals, rbins = self.radial_sampling(
            PA=PA, incl=incl, rbins=rbins, rmin=rmin, rmax=rmax
        )

        if wedge_angle is not None:
            mask_kwargs.update(
                dict(
                    thetamin=wedge_angle,
                    thetamax=180 - wedge_angle,
                    abs_theta=True,
                    exclude_theta=True,
                )
            )
        self.get_mask(PA=PA, incl=incl, **mask_kwargs)

        mask = self.mask.flatten()
        rpnts = r.flatten()[mask]
        toavg = self.data.flatten()[mask]
        ridxs = np.digitize(rpnts, rbins)

        # calculate number of beams per bin
        if assume_correlated:
            if np.diff(rvals).mean() > (self.bmaj + self.bmin) * 0.5:
                nbeams = np.array(
                    [
                        toavg[ridxs == r].size
                        * self.dpix**2
                        / self.Omega_beam_arcsec2
                        for r in range(1, rbins.size)
                    ]
                )
            else:
                ridxs_full = np.digitize(r.flatten(), rbins)
                arc_ratio = np.array(
                    [
                        toavg[ridxs == r].size
                        / self.data.flatten()[ridxs_full == r].size
                        for r in range(1, rbins.size)
                    ]
                )
                # if wedge_angle is None:
                #     arc_length = np.abs(mask_kwargs.get("thetamax", -180) - mask_kwargs.get("thetamin", 180)) / 360.
                #     if mask_kwargs.get("abs_theta", False):
                #         arc_length *= 2
                #     if mask_kwargs.get("exclude_theta", False):
                #         arc_length = 1. - arc_length
                # else:
                #     arc_length = wedge_angle*4 / 360.
                # arc_length = 1. if arc_length > 1. else arc_length
                # print(arc_length)
                from scipy.special import ellipe

                e = np.sin(np.radians(incl))
                nbeams = (
                    4 * rvals * ellipe(e) * arc_ratio / self.bmaj
                )  # elliptic integral for perimeter length of ellipse
                # nbeams = 2.0 * np.pi * rvals * arc_ratio / self.bmaj
        else:
            nbeams = 1

        print("Calculating radial profile...")
        ravgs = np.array([np.mean(toavg[ridxs == r]) for r in range(1, rbins.size)])
        rstds = np.array([np.std(toavg[ridxs == r]) for r in range(1, rbins.size)])
        rstds /= np.sqrt(nbeams)
        print("Done.")

        if save:
            if savefilename is None:
                savefilename = self.fitsname.replace(".fits", "_radialProfile.txt")
                if wedge_angle is not None:
                    savefilename = savefilename.replace(
                        ".txt", "Wedge{}deg.txt".format(wedge_angle)
                    )
            np.savetxt(
                savefilename,
                np.stack([rvals, ravgs, rstds], axis=1),
                fmt="%.8e",
                header=savefileheader,
            )

        return rvals, ravgs, rstds

    def azimuthal_profile(
        self,
        PA=0.0,
        incl=45.0,
        thetabins=None,
        thetamin=-180.0,
        thetamax=180.0,
        r=None,
        rin=0.0,
        rout=1.0,
        assume_correlated=True,
        save=False,
        savefilename=None,
        savefileheader="",
        **mask_kwargs
    ):
        _, theta = self.get_disk_coord(PA=PA, incl=incl)  # in radian

        r_center = np.average([rin, rout]) if r is None else r
        if thetabins is None:
            thetabin_width = (
                self.bmaj / 4.0 / r_center
            )  # 1/4 of bmaj in radian at r_center
            thetabins = np.arange(thetamin, thetamax, thetabin_width)

        tvals = np.average([thetabins[1:], thetabins[:-1]], axis=0)

        if r is not None:
            rin = r - self.bmaj / 8.0
            rout = r + self.bmaj / 8.0

        rmin = mask_kwargs.pop("rmin", rin)
        rmax = mask_kwargs.pop("rmax", rout)
        self.get_mask(PA=PA, incl=incl, rmin=rmin, rmax=rmax, **mask_kwargs)

        mask = self.mask.flatten()
        tpnts = theta.flatten()[mask]
        toavg = self.data.flatten()[mask]
        tidxs = np.digitize(tpnts, thetabins)

        # calculate number of beams per bin
        nbeams = 1
        # if assume_correlated:
        #     if np.diff(tvals).mean() > (self.bmaj + self.bmin) * 0.5:
        #         nbeams = np.array(
        #             [
        #                 toavg[tidxs == t].size
        #                 * self.dpix**2
        #                 / self.Omega_beam_arcsec2
        #                 for t in range(1, thetabins.size)
        #             ]
        #         )
        #     else:
        #         tidxs_full = np.digitize(theta.flatten(), rbins)
        #         arc_ratio = np.array(
        #             [
        #                 toavg[ridxs == r].size
        #                 / self.data.flatten()[ridxs_full == r].size
        #                 for r in range(1, rbins.size)
        #             ]
        #         )
        #         # if wedge_angle is None:
        #         #     arc_length = np.abs(mask_kwargs.get("thetamax", -180) - mask_kwargs.get("thetamin", 180)) / 360.
        #         #     if mask_kwargs.get("abs_theta", False):
        #         #         arc_length *= 2
        #         #     if mask_kwargs.get("exclude_theta", False):
        #         #         arc_length = 1. - arc_length
        #         # else:
        #         #     arc_length = wedge_angle*4 / 360.
        #         # arc_length = 1. if arc_length > 1. else arc_length
        #         # print(arc_length)
        #         from scipy.special import ellipe

        #         e = np.sin(np.radians(incl))
        #         nbeams = (
        #             4 * rvals * ellipe(e) * arc_ratio / self.bmaj
        #         )  # elliptic integral for perimeter length of ellipse
        #         # nbeams = 2.0 * np.pi * rvals * arc_ratio / self.bmaj
        # else:
        #     nbeams = 1

        print("Calculating radial profile...")
        tavgs = np.array([np.mean(toavg[tidxs == t]) for t in range(1, thetabins.size)])
        tstds = np.array([np.std(toavg[tidxs == t]) for t in range(1, thetabins.size)])
        tstds /= np.sqrt(nbeams)
        print("Done.")

        if save:
            if savefilename is None:
                savefilename = self.fitsname.replace(".fits", "_radialProfile.txt")
            np.savetxt(
                savefilename,
                np.stack([tvals, tavgs, tstds], axis=1),
                fmt="%.8e",
                header=savefileheader,
            )

        return tvals, tavgs, tstds

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

    def radial_spectra(
        self,
        PA=0.0,
        incl=45.0,
        rbins=None,
        rmin=0.0,
        rmax=None,
        wedge_angle=None,
        assume_correlated=True,
        save=False,
        savefilename=None,
        savefileheader="",
        **mask_kwargs
    ):
        if self.ndim < 3:
            raise ValueError(
                "The data is not 3D, so calculation of spectra is not possible."
            )

        r, rvals, rbins = self.radial_sampling(
            PA=PA, incl=incl, rbins=rbins, rmin=rmin, rmax=rmax
        )

        if wedge_angle is not None:
            mask_kwargs.update(
                dict(
                    thetamin=wedge_angle,
                    thetamax=180 - wedge_angle,
                    abs_theta=True,
                    exclude_theta=True,
                )
            )

        self.get_mask(PA=PA, incl=incl, **mask_kwargs)

        # _r = np.expand_dims(r.reshape(-1), axis=0) # expand the dimension to be broadcasted with 3D mask array
        # _mask = self.mask.reshape(self.mask.shape[0], -1)
        # _data = self.data.reshape(self.data.shape[0], -1)

        I_arr, dI_arr, npix_arr = [], [], []

        for ridx in range(rvals.size):
            rin = rbins[ridx]
            rout = rbins[ridx + 1]

            I = np.array([np.mean(d[m & (r >= rin) & (r <= rout)]) for d, m in zip(self.data, self.mask)])
            dI = np.array([np.std(d[m & (r >= rin) & (r <= rout)]) for d, m in zip(self.data, self.mask)])
            npix = np.array([d[m & (r >= rin) & (r <= rout)].size for d, m in zip(self.data, self.mask)])

            I_arr.append(I)
            dI_arr.append(dI)
            npix_arr.append(npix)
        
        # calculate number of beams
        if assume_correlated:
            npix = np.squeeze(npix_arr)
            nbeams = self.calc_nbeams(incl=incl, r=r, rbins=rbins, rvals=rvals, npix=npix)
        else:
            nbeams = 1
        
        I = np.squeeze(I_arr)
        dI = np.squeeze(dI_arr) / np.sqrt(nbeams)

        return rvals, self.v, I, dI

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
