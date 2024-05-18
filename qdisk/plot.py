from .utils import jypb_to_K_RJ, jypb_to_K
from .classes import FitsImage, PVFitsImage
from .utils import is_within, plot_2D_map, add_beam
from .model import Keplerian_velocity
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from astropy.visualization import ImageNormalize, LinearStretch
from matplotlib import ticker
import matplotlib.patheffects as pe
import astropy.io.fits as fits
from astropy.wcs import WCS

plt.rcParams.update(
    {
        "figure.figsize": (3.2, 2.4),
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)


class Map(FitsImage):
    def __init__(
        self,
        fitsname_or_data,
        x=None,
        y=None,
        ax=None,
        data_scaling_factor=1.0,
        center_coord=None,
        xlim=None,
        ylim=None,
        downsample=False,
        set_aspect=True,
        invert_xaxis=True,
    ):
        if isinstance(fitsname_or_data, str):
            super().__init__(
                fitsname_or_data,
                squeeze=True,
                rel_dir_ax=True,
                xlim=xlim,
                ylim=ylim,
                downsample=downsample,
            )

        else:
            self.data = fitsname_or_data

            if x is None or y is None:
                raise ValueError("Provide both x and y axes.")

            self.x = x
            self.y = y

        if self.ndim != 2:
            raise ValueError("Image is not in 2D.")

        if center_coord is not None:
            try:
                self.shift_phasecenter_toward(center_coord)
            except AttributeError:
                print("Warning: Phase center shift is not available.")
                pass

        if ax is None:
            fig, ax = plt.subplots(layout="constrained")

        self.ax = ax
        self.data_scaling_factor = data_scaling_factor
        self.center_coord = center_coord
        self.xlim = xlim
        self.ylim = ylim
        self.downsample = downsample

        # axis inversion
        if invert_xaxis:
            self.ax.invert_xaxis()

        self._data_scaling(factor=self.data_scaling_factor)

        if set_aspect:
            self._set_aspect()

    ### DATA UNIT CONVERSION ###

    def convert_unit(self, unit="K", nu=None, RJ_approx=False):
        ### assume the original data unit is Jy / beam or mJy / beam
        if not np.isnan(self.restfreq):
            nu = self.restfreq
        elif nu is None:
            raise ValueError(
                "Rest frequency not found. Please provide it via *nu* argument."
            )

        data = self.data / self.data_scaling_factor  # restore original unit
        if "mJy" in self.data_unit:
            data *= 1e-3  # in Jy /beam

        if unit == "K":
            if RJ_approx:
                self.data = jypb_to_K_RJ(data, nu, self.beam[:2])
            else:
                self.data = jypb_to_K(data, nu, self.beam[:2])

    ### MASKING ###

    def mask(
        self,
        rmin=None,
        rmax=None,
        x0=0.0,
        y0=0.0,
        PA=0.0,
        incl=0.0,
        vmin=None,
        vmax=None,
        exclude_v=False,
        threshold=None,
        rms=None,
        user_mask=None,
        combine="and",
        apply=True,
        fill=np.nan,
    ):
        mask_list = []

        # spatial mask
        if (rmin is not None) or (rmax is not None):
            mask_list.append(
                self._get_annulus_mask(
                    rmin=rmin, rmax=rmax, x0=x0, y0=y0, PA=PA, incl=incl
                )
            )

        # value mask
        if (vmin is not None) or (vmax is not None):
            mask_list.append(
                self._get_value_mask(vmin=vmin, vmax=vmax, exclude=exclude_v)
            )

        # threshold mask
        if threshold is not None:
            mask_list.append(
                self._get_sigma_threshold_mask(threshold=threshold, rms=rms)
            )

        # user mask
        if user_mask is not None:
            mask_list.append(user_mask.astype(bool))

        # mask conbination
        combine = getattr(np, "logical_" + combine)
        mask = combine.reduce(np.array(mask_list), axis=0)

        if apply:
            self.data[~mask] = fill

        return mask

    def _get_annulus_mask(self, rmax=None, rmin=None, x0=0.0, y0=0.0, PA=0.0, incl=0.0):
        r, _ = self.get_disk_coord(x0=x0, y0=y0, PA=PA, incl=incl, frame="polar")
        rmin = np.nanmin(r) if rmin is None else rmin
        rmax = np.nanmax(r) if rmax is None else rmax
        return np.logical_and(r >= rmin, r <= rmax)

    def _get_value_mask(self, vmin=None, vmax=None, exclude=False):
        vmin = np.nanmin(self.data) if vmin is None else vmin
        vmax = np.nanmax(self.data) if vmax is None else vmax
        mask = np.logical_and(self.data >= vmin, self.data <= vmax)
        mask = ~mask if exclude else mask
        return mask

    def _get_sigma_threshold_mask(self, threshold, rms=None):
        """Only include values larger than threshold * rms"""
        if rms is None:
            rms = self.rms
        return self._get_value_mask(vmin=threshold * rms)

    ### MAP FUNCTION ###

    def plot_colormap(
        self,
        method="pcolorfast",
        cmap="viridis",
        vmin=None,
        vmax=None,
        interval=None,
        stretch=None,
        **kwargs
    ):
        if not method in ["imshow", "pcolorfast", "pcolorfast_wcs", "pcolormesh", "contourf", "pcolormesh_wcs", "contourf_wcs"]:
            raise AttributeError(
                "Method {:s} is not supported for colormap plot.".format(method)
            )

        # normalization
        if stretch is None:
            stretch = LinearStretch()
        norm = self._normalize(vmin=vmin, vmax=vmax, interval=interval, stretch=stretch)
        plot = getattr(self, "_" + method + "_self")
        self.colormap = plot(cmap=cmap, norm=norm, **kwargs)

    def overlay_contour(
        self,
        fitsname_or_data="self",
        x=None,
        y=None,
        data_scaling_factor=1.0,
        levels=5,
        color="grey",
        linewidth=1.0,
        linestyle="solid",
        **kwargs
    ):

        if isinstance(fitsname_or_data, str):
            if fitsname_or_data == "self":
                x = self.x
                y = self.y
                data = self.data
            else:
                image = FitsImage(
                    fitsname=fitsname_or_data,
                    squeeze=True,
                    rel_dir_ax=True,
                    xlim=self.xlim,
                    ylim=self.ylim,
                    downsample=self.downsample,
                )
                image.shift_phasecenter_toward(self.center_coord)
                x = image.x
                y = image.y
                data = image.data * data_scaling_factor
        else:
            x = self.x if x is None else x
            y = self.y if y is None else y
            data = fitsname_or_data * data_scaling_factor

        self.contour = self._contour(
            x,
            y,
            data,
            levels=levels,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            **kwargs
        )

    def _contour(
        self,
        x,
        y,
        data,
        levels=5,
        color="black",
        linewidth=1.0,
        linestyle="solid",
        **kwargs
    ):
        im = self.ax.contour(
            x,
            y,
            data,
            levels=levels,
            colors=color,
            linewidths=linewidth,
            linestyles=linestyle,
            **kwargs
        )
        return im

    def _contour_self(self, levels=5, color="black"):
        im = self.ax.contour(self.x, self.y, self.data, levels=levels, colors=color)
        return im

    # def _set_contourf_extend(self):

    def _contourf_self(self, levels=10, cmap="viridis", norm=None, **kwargs):
        if isinstance(levels, (list, np.ndarray)):
            vmin = np.nanmin(levels)
            vmax = np.nanmax(levels)
        else:
            vmin, vmax = norm.vmin, norm.vmax
        data = self.data.copy()
        # data[data < vmin] = vmin
        # data[data > vmax] = vmax
        im = self.ax.contourf(
            self.x, self.y, data, levels=levels, cmap=cmap, norm=norm, **kwargs
        )
        return im

    def _pcolorfast_self(self, cmap="viridis", norm=None, **kwargs):
        im = self.ax.pcolorfast(
            self.x, self.y, self.data, rasterized=True, cmap=cmap, norm=norm, **kwargs
        )
        return im

    def _pcolormesh_self(self, cmap="viridis", norm=None, **kwargs):
        im = self.ax.pcolormesh(
            self.x, self.y, self.data, rasterized=True, cmap=cmap, norm=norm, **kwargs
        )
        return im

    def _normalize(self, vmin=None, vmax=None, interval=None, stretch=LinearStretch()):
        norm = ImageNormalize(
            self.data, vmin=vmin, vmax=vmax, interval=interval, stretch=stretch
        )
        return norm

    def _data_scaling(self, factor):
        self.data *= factor

    ### FANCY ADDENDA STUFF ###

    def add_beam(self, beam=None, loc="lower left", color="white", fill=True, hatch="///////////"):
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse

        if beam is not None:
            bmaj, bmin, pa = beam
            width = bmaj
            height = bmin
            angle = 90 - pa
        else:
            width = self.bmaj
            height = self.bmin
            angle = 90 - self.bpa  # to make measured from east
        beam = AnchoredEllipse(
            self.ax.transData,
            width=width,
            height=height,
            angle=angle,
            loc=loc,
            pad=0.5,
            borderpad=0.5,
            frameon=False,
        )
        beam.ellipse.set(color=color, fill=fill, hatch=hatch)
        self.ax.add_artist(beam)

    def add_scalebar(
        self, scale=50, text=None, width=0.0, loc="lower right", color="white", **kwargs
    ):

        scalebar = self._scalebar_with_label(
            scale=scale, text=text, width=width, loc=loc, color=color, **kwargs
        )
        self.ax.add_artist(scalebar)

    def _scalebar_with_label(
        self, scale, text, width=0.0, loc="lower right", color="white", **kwargs
    ):
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        import matplotlib.font_manager as fm

        scalebar = AnchoredSizeBar(
            self.ax.transData,
            size=scale,
            label=text,
            loc=loc,
            pad=0.1,
            borderpad=0.5,
            sep=3,
            frameon=False,
            color=color,
            size_vertical=width,
            fontproperties=fm.FontProperties(size=9),
            **kwargs
        )

        return scalebar

    ### COLORBAR STUFF ###

    def _get_colormap_normalize(self):
        return self.colormap.norm

    def _set_colorbar_extend(self):
        norm = self._get_colormap_normalize()
        extend = "neither"

        if np.nanmax(self.data) > norm.vmax:
            extend = "max"
            if np.nanmin(self.data) < norm.vmin:
                extend = "both"
                return extend

        if np.nanmin(self.data) < norm.vmin:
            extend = "min"

        return extend

    def add_colorbar(
        self,
        cax=None,
        position="right",
        size="5%",
        pad=0.1,
        label=None,
        rotation=270,
        labelpad=15,
        **kwargs
    ):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.axes as maxes

        if cax is None:
            divider = make_axes_locatable(self.ax)
            cax = divider.append_axes(
                position=position, size=size, axes_class=maxes.Axes, pad=pad
            )

        extend = self._set_colorbar_extend()

        fig = self.ax.get_figure()
        orientation = (
            "horizontal" if position == "top" or position == "bottom" else "vertical"
        )
        self.colorbar = fig.colorbar(
            self.colormap, cax=cax, orientation=orientation, extend=extend, **kwargs
        )
        self.colorbar.set_label(label, rotation=rotation, labelpad=labelpad)

        if position == "top":
            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
        elif position == "bottom":
            cax.xaxis.set_ticks_position("bottom")
            cax.xaxis.set_label_position("bottom")

        return cax

    ### APPEARANCE ###

    def _set_aspect(self):
        self.ax.set_aspect(1.0 / self.ax.get_data_ratio())

    def set_aspect(self, aspect="equal"):
        """Wrapper of ax.set_aspect()"""

        if aspect == "equal":
            aspect = 1.0

        self.ax.set_aspect(aspect / self.ax.get_data_ratio())

    def _set_xlabel(self, xlabel):
        self.ax.set_xlabel(xlabel)

    def _set_ylabel(self, ylabel):
        self.ax.set_ylabel(ylabel)

    def set_labels(self, xlabel=None, ylabel=None):
        self._set_xlabel(xlabel)
        self._set_ylabel(ylabel)

    def _get_major_ticker_locator(self, nticks=None, interval=None):
        if nticks is not None:
            return ticker.MaxNLocator(nticks)
        if interval is not None:
            return ticker.MultipleLocator(interval)
        return ticker.AutoLocator()

    def _get_minor_ticker_locator(self, nticks=None, interval=None):
        if nticks is not None:
            return ticker.MaxNLocator(nticks)
        if interval is not None:
            return ticker.MultipleLocator(interval)
        return ticker.AutoMinorLocator()

    def set_xaxis_ticker(
        self,
        major=True,
        minor=False,
        majornticks=None,
        minornticks=None,
        majorinterval=None,
        minorinterval=None,
    ):
        if not major:
            majorlocator = ticker.NullLocator()
        else:
            majorlocator = self._get_major_ticker_locator(
                nticks=majornticks, interval=majorinterval
            )

        if not minor:
            minorlocator = ticker.NullLocator()
        else:
            minorlocator = self._get_minor_ticker_locator(
                nticks=minornticks, interval=minorinterval
            )

        self.ax.xaxis.set_major_locator(majorlocator)
        self.ax.xaxis.set_minor_locator(minorlocator)

    def set_yaxis_ticker(
        self,
        major=True,
        minor=False,
        majornticks=None,
        minornticks=None,
        majorinterval=None,
        minorinterval=None,
    ):
        if not major:
            majorlocator = ticker.NullLocator()
        else:
            majorlocator = self._get_major_ticker_locator(
                nticks=majornticks, interval=majorinterval
            )

        if not minor:
            minorlocator = ticker.NullLocator()
        else:
            minorlocator = self._get_minor_ticker_locator(
                nticks=minornticks, interval=minorinterval
            )

        self.ax.yaxis.set_major_locator(majorlocator)
        self.ax.yaxis.set_minor_locator(minorlocator)

    def set_ticker(
        self,
        major=True,
        minor=False,
        majornticks=None,
        minornticks=None,
        majorinterval=None,
        minorinterval=None,
    ):
        self.set_xaxis_ticker(
            major=major,
            minor=minor,
            majornticks=majornticks,
            minornticks=minornticks,
            majorinterval=majorinterval,
            minorinterval=minorinterval,
        )
        self.set_yaxis_ticker(
            major=major,
            minor=minor,
            majornticks=majornticks,
            minornticks=minornticks,
            majorinterval=majorinterval,
            minorinterval=minorinterval,
        )

        if minor:
            self.ax.minorticks_on()

    # def _scalebar_without_label(self, scale, width=0.0, loc="lower right", color="white"):
    #     from matplotlib.patches import Rectangle

    #     scalebar = Rectangle()

    # def set_xlim(self):

    # def _get_threshold_mask(self, ):
    #     self.estimate_rms()
            

class WCSMap(Map):
    def __init__(
        self,
        fitsname,
        ax=None, # should be wcsaxes
        data_scaling_factor=1.0,
        center_coord=None,
        xlim=None,
        ylim=None,
        downsample=False,
        set_aspect=True,
    ):
        self.wcs = WCS(fits.getheader(fitsname)).celestial
        if ax is None:
            fig, ax = plt.subplots(layout="constrained", projection=self.wcs)
        else:
            super().__init__(
                fitsname_or_data=fitsname,
                ax=ax,
                data_scaling_factor=data_scaling_factor,
                center_coord=center_coord,
                xlim=xlim,
                ylim=ylim,
                downsample=downsample,
                set_aspect=set_aspect,
                invert_xaxis=False
            )

    ### MAP FUNCTION ###

    def plot_colormap(
        self,
        method="pcolorfast",
        cmap="viridis",
        vmin=None,
        vmax=None,
        interval=None,
        stretch=None,
        **kwargs
    ):
        if not method in ["imshow", "pcolorfast", "pcolormesh", "contourf"]:
            raise AttributeError(
                "Method {:s} is not supported for colormap plot.".format(method)
            )

        # normalization
        if stretch is None:
            stretch = LinearStretch()
        norm = self._normalize(vmin=vmin, vmax=vmax, interval=interval, stretch=stretch)
        plot = getattr(self, "_" + method + "_self")
        self.colormap = plot(cmap=cmap, norm=norm, **kwargs)

    def overlay_contour(
        self,
        fitsname_or_data="self",
        data_scaling_factor=1.0,
        levels=5,
        color="grey",
        linewidth=1.0,
        linestyle="solid",
        **kwargs
    ):

        if isinstance(fitsname_or_data, str):
            if fitsname_or_data == "self":
                data = self.data
            else:
                image = FitsImage(
                    fitsname=fitsname_or_data,
                    squeeze=True,
                    xlim=self.xlim,
                    ylim=self.ylim,
                    downsample=self.downsample,
                )
                image.shift_phasecenter_toward(self.center_coord)
                data = image.data * data_scaling_factor
        else:
            data = fitsname_or_data * data_scaling_factor

        self.contour = self._contour(
            data,
            levels=levels,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            **kwargs
        )

    def _contour(
        self,
        data,
        levels=5,
        color="black",
        linewidth=1.0,
        linestyle="solid",
        **kwargs
    ):
        im = self.ax.contour(
            data,
            levels=levels,
            colors=color,
            linewidths=linewidth,
            linestyles=linestyle,
            **kwargs
        )
        return im

    def _contour_self(self, levels=5, color="black"):
        im = self.ax.contour(self.data, levels=levels, colors=color)
        return im

    def _contourf_self(self, levels=10, cmap="viridis", norm=None, **kwargs):
        if isinstance(levels, (list, np.ndarray)):
            vmin = np.nanmin(levels)
            vmax = np.nanmax(levels)
        else:
            vmin, vmax = norm.vmin, norm.vmax
        data = self.data.copy()
        # data[data < vmin] = vmin
        # data[data > vmax] = vmax
        im = self.ax.contourf(
            data, levels=levels, cmap=cmap, norm=norm, **kwargs
        )
        return im

    def _pcolorfast_self(self, cmap="viridis", norm=None, **kwargs):
        im = self.ax.pcolorfast(
            self.data, rasterized=True, cmap=cmap, norm=norm, **kwargs
        )
        return im

    def _pcolormesh_self(self, cmap="viridis", norm=None, **kwargs):
        im = self.ax.pcolormesh(
            self.data, rasterized=True, cmap=cmap, norm=norm, **kwargs
        )
        return im

    ### FANCY ADDENDA STUFF ###

    def add_beam(self, beam=None, loc="lower left", color="white", fill=True, hatch="///////////"):
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse

        if beam is not None:
            bmaj, bmin, pa = beam
            width = bmaj
            height = bmin
            angle = 90 - pa
        else:
            width = self.bmaj
            height = self.bmin
            angle = 90 - self.bpa  # to make measured from east
        beam = AnchoredEllipse(
            self.ax.transData,
            width=width / self.dpix,
            height=height / self.dpix,
            angle=angle,
            loc=loc,
            pad=0.5,
            borderpad=0.5,
            frameon=False,
        )
        beam.ellipse.set(color=color, fill=fill, hatch=hatch)
        self.ax.add_artist(beam)

    def add_scalebar(
        self, scale=50, text=None, width=0.0, loc="lower right", color="white", **kwargs
    ):

        scalebar = self._scalebar_with_label(
            scale=scale, text=text, width=width, loc=loc, color=color, **kwargs
        )
        self.ax.add_artist(scalebar)

    def _scalebar_with_label(
        self, scale, text, width=0.0, loc="lower right", color="white", **kwargs
    ):
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        import matplotlib.font_manager as fm

        scalebar = AnchoredSizeBar(
            self.ax.transData,
            size=scale / self.dpix,
            label=text,
            loc=loc,
            pad=0.1,
            borderpad=0.5,
            sep=3,
            frameon=False,
            color=color,
            size_vertical=width,
            fontproperties=fm.FontProperties(size=9),
            **kwargs
        )

        return scalebar

    

class MultiPlot:
    def __init__(self, ncols=None, nrows=None, npanels=None, figsize=(3,3), sharex=True, sharey=True, max_figsize=(15, None)):
        if (ncols is None) and (nrows is None):
            assert npanels is not None, "Provide either ncols and nrows or npanels."
            self.npanels = npanels
            self.ncols, self.nrows = self.get_ncols_nrows(npanels=npanels, figsize=figsize, max_figsize=max_figsize)
        else:
            self.ncols = ncols
            self.nrows = nrows
            self.npanels = npanels if npanels is not None else self.ncols * self.nrows
        
        self.width_each, self.height_each = figsize
        self.width = self.ncols * self.width_each
        self.height = self.nrows * self.height_each
        self.fig, self.axes = plt.subplots(
            figsize=(self.width, self.height),
            nrows=self.nrows,
            ncols=self.ncols,
            sharex=sharex,
            sharey=sharey,
            constrained_layout=True,
        )
        self.axes = self.axes if isinstance(self.axes, np.ndarray) else np.array([self.axes])

        # clear the extra axes
        for i in range(self.npanel, len(self.axes.flatten())):
            ax = self.axes.flatten()[i]
            ax.set_axis_off()

    @staticmethod
    def get_ncols_nrows(npanels, figsize, max_figsize):
        nrows = int(npanels**0.5)
        ncols = int(npanels / nrows * 0.999) + 1

        _width, _height = figsize
        width = ncols * _width
        height = nrows * _height

        width_max, height_max = max_figsize

        if (width_max is not None) and width > width_max:
            ncols = int(width_max / _width)
            nrows = int(npanels / ncols * 0.999) + 1
        if (height_max is not None) and (height > height_max):
            nrows = int(height_max / _height)
            ncols = int(npanels / nrows * 0.999) + 1
        return ncols, nrows
    
    def get_lower_left_axis_index(self):
        return int((self.nrows - 1) * self.ncols)
    
    def get_lower_left_axis(self):
        return self.axes.flatten()[self.get_lower_left_axis_index()]

    def set_labels(self, xlabel=None, ylabel=None):
        ax = self.get_lower_left_axis()
        ax.set(xlabel=xlabel, ylabel=ylabel)

"""
class ChannelMap(FitsImage):
    def __init__(
        self,
        fitsname_or_data,
        x=None,
        y=None,
        v=None,
        data_scaling_factor=1.0,
        center_coord=None,
        rel_dir_ax=True,
        xlim=None,
        ylim=None,
        vlim=None,
        nu0=None,
        downsample=False,
        set_aspect=True
    ):
        if isinstance(fitsname_or_data, str):
            super().__init__(
                fitsname_or_data,
                rel_dir_ax=rel_dir_ax,
                xlim=xlim,
                ylim=ylim,
                vlim=vlim,
                nu0=nu0,
                downsample=downsample,
                skipdata=False,
            )
        else:
            self.data = fitsname_or_data

            if x is None or y is None:
                raise ValueError("Provide both x and y axes.")
            if v is None:
                raise ValueError("Provide velocity axis.")

            self.x = x
            self.y = y
            self.v = v

        if self.ndim != 3:
            raise ValueError("Image is not in 3D.")

        if center_coord is not None:
            try:
                self.shift_phasecenter_toward(center_coord)
            except AttributeError:
                print("Warning: Phase center shift is not available.")
                pass
        
        self.figure = MultiPlot(npanels=self.v.size, sharex=True, sharey=True, figsize=(2,2))

        self.axes = self.figure.axes.flatten()
        self.data_scaling_factor = data_scaling_factor
        self.center_coord = center_coord
        self.xlim = xlim
        self.ylim = ylim
        self.vlim = vlim
        self.nu0 = nu0
        self.downsample = downsample

        # set the Map class for each panel
        self.maps = []
        for i, ax in enumerate(self.axes):
            im = Map(
                fitsname_or_data=self.data[i],
                x=self.x,
                y=self.y,
                ax=ax,
                data_scaling_factor=self.data_scaling_factor,
            )
            if set_aspect:
                im._set_aspect()
            self.maps.append(im)

    ### MAP FUNCTION ###

    def plot_colormap(
        self,
        method="pcolorfast",
        cmap="viridis",
        vmin=None,
        vmax=None,
        normalize_each_chan=False,
        interval=None,
        stretch=None,
        **kwargs
    ):
        if not normalize_each_chan:
            vmin = np.min(np.isfinite(self.data))
            vmax = np.max(np.isfinite(self.data))
    
        for map in self.maps:
            map.plot_colormap(
                method=method,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interval=interval,
                stretch=stretch,
                **kwargs
            )

    def overlay_contour(
        self,
        fitsname_or_data="self",
        x=None,
        y=None,
        data_scaling_factor=1.0,
        levels=5,
        color="grey",
        linewidth=1.0,
        linestyle="solid",
        **kwargs
    ):
        for map in self.maps:
            map.overlay_contour(
                fitsname_or_data=fitsname_or_data,
                x=x,
                y=y,
                data_scaling_factor=data_scaling_factor,
                levels=levels,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                **kwargs
            )

    ### FANCY ADDENDA STUFF ###

    def add_beam(self, panel="lower left", loc="lower left", color="white", fill=True, hatch="///////////"):
        if panel == "lower left":
            index = self.figure.get_lower_left_axis_index()
            self.maps[index].add_beam(loc=loc, color=color, fill=fill, hatch=hatch)
        elif panel == "all":
            for map in self.maps:
                map.add_beam(loc=loc, color=color, fill=fill, hatch=hatch)

    def add_scalebar(
        self, scale=50, text=None, width=0.0, panel="lower left", loc="lower right", color="white", **kwargs
    ):
        if panel == "lower left":
            index = self.figure.get_lower_left_axis_index()
            self.maps[index].add_scalebar(scale=scale, text=text, width=width, loc=loc, color=color, **kwargs)
        elif panel == "all":
            for map in self.maps:
                map.add_scalebar(scale=scale, text=text, width=width, loc=loc, color=color, **kwargs)

    ### COLORBAR STUFF ###

    def add_colorbar(
        self,
        cax=None,
        position="right",
        size="5%",
        pad=0.1,
        label=None,
        rotation=270,
        labelpad=15,
    ):
        map = self.maps[-1]
        map.add_colorbar(cax=cax, position=position, size=size, pad=pad, label=label, rotation=rotation, labelpad=labelpad)
    
    ### APPERANCE ###

    def _set_aspect(self):
        self.ax.set_aspect(1.0 / self.ax.get_data_ratio())
"""



class PVDiagram(PVFitsImage):
    def __init__(
        self,
        fitsname_or_data,
        posax=None,
        velax=None,
        ax=None,
        data_scaling_factor=1.0,
        xlim=None,
        vlim=None,
        downsample=False,
    ):

        if isinstance(fitsname_or_data, str):
            super().__init__(
                fitsname=fitsname_or_data, xlim=xlim, vlim=vlim, downsample=downsample
            )
        else:
            self.data = fitsname_or_data
            self.posax = posax
            self.v = velax

        if self.ndim != 2:
            raise ValueError("Image is not in 2D.")

        if ax is None:
            fig, ax = plt.subplots()

        self.ax = ax
        self.data_scaling_factor = data_scaling_factor
        self.xlim = xlim
        self.vlim = vlim
        self.downsample = downsample

        self._data_scaling(factor=self.data_scaling_factor)

    def convert_unit(self, unit="K", nu=None, RJ_approx=False):
        ### assume the original data unit is Jy / beam or mJy / beam
        if not np.isnan(self.restfreq):
            nu = self.restfreq
        elif nu is None:
            raise ValueError(
                "Rest frequency not found. Please provide it via *nu* argument."
            )

        data = self.data / self.data_scaling_factor  # restore original unit
        if "mJy" in self.data_unit:
            data *= 1e-3  # in Jy /beam

        if unit == "K":
            if RJ_approx:
                self.data = jypb_to_K_RJ(data, nu, self.beam[:2])
            else:
                self.data = jypb_to_K(data, nu, self.beam[:2])

    def plot_colormap(
        self,
        method="pcolorfast",
        cmap="viridis",
        vmin=None,
        vmax=None,
        interval=None,
        stretch=None,
    ):
        if not method in ["imshow", "pcolorfast", "pcolormesh"]:
            raise AttributeError(
                "Method {:s} is not supported for colormap plot.".format(method)
            )

        # normalization
        if stretch is None:
            stretch = LinearStretch()
        norm = self._normalize(vmin=vmin, vmax=vmax, interval=interval, stretch=stretch)
        plot = getattr(self, "_" + method + "_self")
        self.colormap = plot(cmap=cmap, norm=norm)

    def overlay_contour(
        self,
        fitsname_or_data="self",
        data_scaling_factor=1.0,
        levels=5,
        color="grey",
        linewidth=1.0,
        linestyle="solid",
        **kwargs
    ):

        if fitsname_or_data == "self":
            x = self.posax
            v = self.v
            data = self.data
        elif isinstance(fitsname_or_data, str):
            image = PVFitsImage(
                fitsname=fitsname_or_data,
                squeeze=True,
                xlim=self.xlim,
                vlim=self.vlim,
                downsample=self.downsample,
            )
            x = image.posax
            v = image.v
            data = image.data * data_scaling_factor
        else:
            data = fitsname_or_data * data_scaling_factor

        self.contour = self._contour(
            x,
            v,
            data,
            levels=levels,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            **kwargs
        )

    def _contour(
        self,
        x,
        v,
        data,
        levels=5,
        color="black",
        linewidth=1.0,
        linestyle="solid",
        **kwargs
    ):
        im = self.ax.contour(
            x,
            v,
            data,
            levels=levels,
            colors=color,
            linewidths=linewidth,
            linestyles=linestyle,
            **kwargs
        )
        return im

    def _contour_self(self, levels=5, color="black"):
        im = self.ax.contour(self.posax, self.v, self.data, levels=levels, colors=color)
        return im

    def _pcolorfast_self(self, cmap="viridis", norm=None):
        im = self.ax.pcolorfast(
            self.posax, self.v, self.data, rasterized=True, cmap=cmap, norm=norm
        )
        return im

    def _normalize(self, vmin=None, vmax=None, interval=None, stretch=LinearStretch()):
        norm = ImageNormalize(
            self.data, vmin=vmin, vmax=vmax, interval=interval, stretch=stretch
        )
        return norm

    def _data_scaling(self, factor):
        self.data *= factor

    def _get_colormap_normalize(self):
        return self.colormap.norm

    def _set_colorbar_extend(self):
        norm = self._get_colormap_normalize()
        extend = "neither"

        if np.nanmax(self.data) > norm.vmax:
            extend = "max"
            if np.nanmin(self.data) < norm.vmin:
                extend = "both"
                return extend

        if np.nanmin(self.data) < norm.vmin:
            extend = "min"

        return extend

    def add_Keplerian_curve(
        self,
        Mstar=1.0,
        distance=100.0,
        incl=90.0,
        vsys=0.0,
        rotation_sense=1.0,
        **kwargs
    ):

        # store the original ylim to avoid too large plot region along y axis
        ylim = self.ax.get_ylim()

        # default color
        color = kwargs.pop("color", "white")

        if isinstance(Mstar, float):
            # calculate the Keplerian velocity at each positional point
            vkep = Keplerian_velocity(
                r=self.posax * distance, Mstar=Mstar, incl=rotation_sense * incl
            )

            # plot
            self.ax.plot(
                self.posax[self.posax > 0],
                -vkep[self.posax > 0] + vsys,
                color=color,
                **kwargs
            )
            self.ax.plot(
                self.posax[self.posax < 0],
                vkep[self.posax < 0] + vsys,
                color=color,
                **kwargs
            )

        else:
            # calculate the Keplerian velocity for each element of Mstar
            vkep_l = Keplerian_velocity(
                r=self.posax * distance, Mstar=Mstar[0], incl=rotation_sense * incl
            )
            vkep_u = Keplerian_velocity(
                r=self.posax * distance, Mstar=Mstar[1], incl=rotation_sense * incl
            )

            # plot
            self.ax.fill_between(
                self.posax[self.posax > 0],
                -vkep_u[self.posax > 0] + vsys,
                -vkep_l[self.posax > 0] + vsys,
                color=color,
                **kwargs
            )
            self.ax.fill_between(
                self.posax[self.posax < 0],
                vkep_l[self.posax < 0] + vsys,
                vkep_u[self.posax < 0] + vsys,
                color=color,
                **kwargs
            )

        # set original ylim
        self.ax.set_ylim(ylim)

    def add_colorbar(
        self,
        position="right",
        size="5%",
        pad=0.1,
        label=None,
        rotation=270,
        labelpad=10,
    ):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.axes as maxes

        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes(
            position=position, size=size, axes_class=maxes.Axes, pad=pad
        )
        extend = self._set_colorbar_extend()

        fig = self.ax.get_figure()
        self.colorbar = fig.colorbar(self.colormap, cax=cax, extend=extend)
        self.colorbar.set_label(label, rotation=rotation, labelpad=labelpad)

    ### APPEARANCE ###

    def _set_aspect(self):
        self.ax.set_aspect(1.0 / self.ax.get_data_ratio())

    def set_aspect(self, aspect="equal"):
        """Wrapper of ax.set_aspect()"""

        if aspect == "equal":
            aspect = 1.0

        self.ax.set_aspect(aspect / self.ax.get_data_ratio())

    def _set_xlabel(self, xlabel):
        self.ax.set_xlabel(xlabel)

    def _set_ylabel(self, ylabel):
        self.ax.set_ylabel(ylabel)

    def set_labels(self, xlabel=None, ylabel=None):
        self._set_xlabel(xlabel)
        self._set_ylabel(ylabel)

    def _get_major_ticker_locator(self, nticks=None, interval=None):
        if nticks is not None:
            return ticker.MaxNLocator(nticks)
        if interval is not None:
            return ticker.MultipleLocator(interval)
        return ticker.AutoLocator()

    def _get_minor_ticker_locator(self, nticks=None, interval=None):
        if nticks is not None:
            return ticker.MaxNLocator(nticks)
        if interval is not None:
            return ticker.MultipleLocator(interval)
        return ticker.AutoMinorLocator()

    def set_xaxis_ticker(
        self,
        major=True,
        minor=False,
        majornticks=None,
        minornticks=None,
        majorinterval=None,
        minorinterval=None,
    ):
        if not major:
            majorlocator = ticker.NullLocator()
        else:
            majorlocator = self._get_major_ticker_locator(
                nticks=majornticks, interval=majorinterval
            )

        if not minor:
            minorlocator = ticker.NullLocator()
        else:
            minorlocator = self._get_minor_ticker_locator(
                nticks=minornticks, interval=minorinterval
            )

        self.ax.xaxis.set_major_locator(majorlocator)
        self.ax.xaxis.set_minor_locator(minorlocator)

    def set_yaxis_ticker(
        self,
        major=True,
        minor=False,
        majornticks=None,
        minornticks=None,
        majorinterval=None,
        minorinterval=None,
    ):
        if not major:
            majorlocator = ticker.NullLocator()
        else:
            majorlocator = self._get_major_ticker_locator(
                nticks=majornticks, interval=majorinterval
            )

        if not minor:
            minorlocator = ticker.NullLocator()
        else:
            minorlocator = self._get_minor_ticker_locator(
                nticks=minornticks, interval=minorinterval
            )

        self.ax.yaxis.set_major_locator(majorlocator)
        self.ax.yaxis.set_minor_locator(minorlocator)

    def set_ticker(
        self,
        major=True,
        minor=False,
        majornticks=None,
        minornticks=None,
        majorinterval=None,
        minorinterval=None,
    ):
        self.set_xaxis_ticker(
            major=major,
            minor=minor,
            majornticks=majornticks,
            minornticks=minornticks,
            majorinterval=majorinterval,
            minorinterval=minorinterval,
        )
        self.set_yaxis_ticker(
            major=major,
            minor=minor,
            majornticks=majornticks,
            minornticks=minornticks,
            majorinterval=majorinterval,
            minorinterval=minorinterval,
        )

        if minor:
            self.ax.minorticks_on()


class ChannelMap(FitsImage):
    def __init__(
        self,
        fitsname_or_data,
        x=None,
        y=None,
        v=None,
        data_scaling_factor=1,
        center_coord=None,
        xlim=None,
        ylim=None,
        vlim=None,
        nu0=None,
        downsample=False,
    ):

        if isinstance(fitsname_or_data, str):
            super().__init__(
                fitsname_or_data,
                squeeze=True,
                rel_dir_ax=True,
                xlim=xlim,
                ylim=ylim,
                vlim=vlim,
                nu0=nu0,
                downsample=downsample,
            )

        else:
            self.data = fitsname_or_data
            # self.ndim = self.data.ndim

            if x is None or y is None:
                raise ValueError("Provide both x axis and y axis.")
            if v is None:
                raise ValueError("Provide velocity axis.")
            self.x = x
            self.y = y
            self.v = v
            # self.nchan = v.size

        if self.ndim < 3:
            raise ValueError("The image is 2D.")

        self.center_coord = center_coord
        if self.center_coord is not None:
            try:
                self.shift_phasecenter_toward(self.center_coord)
            except AttributeError:
                print("Warning: Phase center shift is not available.")
                pass
        
        if np.all(np.diff(self.v) < 0):
            self.v = self.v[::-1]
            self.data = self.data[::-1]

        self._data_scaling(factor=data_scaling_factor)

    # def _get_nrows_ncols(self):
    #     ncols = np.ceil(self.nchan**0.5).astype(int)
    #     nrows = np.ceil(self.nchan / ncols).astype(int)
    #     return nrows, ncols
    
    @staticmethod
    def get_nrows_ncols(npanels, ncols=None, nrows=None, panelsize=(2.0, 2.0), max_figsize=(12, None)):
        if (ncols is None) and (nrows is None):
            nrows = int(npanels**0.5)
            ncols = int(npanels / nrows * 0.999) + 1
        elif (ncols is not None) and (nrows is None):
            nrows = int(npanels / ncols * 0.999) + 1
        elif (ncols is None) and (nrows is not None):
            ncols = int(npanels / nrows * 0.999) + 1


        _width, _height = panelsize
        width = ncols * _width
        height = nrows * _height

        width_max, height_max = max_figsize

        if (width_max is not None) and width > width_max:
            ncols = int(width_max / _width)
            nrows = int(npanels / ncols * 0.999) + 1
        if (height_max is not None) and (height > height_max):
            nrows = int(height_max / _height)
            ncols = int(npanels / nrows * 0.999) + 1
        return nrows, ncols

    # def _get_figsize(self):
    #     return (self.ncols * 3, self.nrows * 3)

    def set_imagegrid(self, ncols=None, nrows=None, panelsize=(2.0, 2.0), figsize=None, max_figsize=(12, None), pad=0.1, cbar_mode="bottom right", cbar_label=None):
        # setup figure instance
        self.nrows, self.ncols = self.get_nrows_ncols(npanels=self.v.size, ncols=ncols, nrows=nrows, panelsize=panelsize, max_figsize=max_figsize)
        w, h = panelsize
        self.fig = plt.figure(figsize=(self.ncols*w, self.nrows*h) if figsize is None else figsize, layout="constrained")

        self.colorbar = cbar_mode is not None
        self.imgrid = ImageGrid(
            self.fig,
            rect=111,
            nrows_ncols=(self.nrows, self.ncols),
            share_all=True,
            axes_pad=pad,
            cbar_mode="edge" if cbar_mode == "bottom right" else cbar_mode,
            cbar_location="right",
        )
        ### ImageGrid should have param of *ngrids*, but specifying this param cause an error (maybe bug?).
        ### Here is workaround for that, removing axes on which no data are drawn.
        for i in range(self.nchan, len(self.imgrid)):
            self.imgrid[i].set_axis_off()

        if cbar_mode == "bottom right":
            for ax in self.imgrid.cbar_axes[: self.nrows - 1]:
                ax.set_visible(False)
            self.imgrid.cbar_axes = [
                self.imgrid.cbar_axes[self.nrows - 1]
            ]  # only colorbar at bottom right panel

        self.cbar_label = cbar_label if cbar_label is not None else self.data_unit

    def _data_scaling(self, factor):
        self.data *= factor

    def _set_colorbar_extend(self):
        extend = "neither"

        if np.nanmax(self.data) > self.norm.vmax:
            extend = "max"
            if np.nanmin(self.data) < self.norm.vmin:
                extend = "both"
                return extend

        if np.nanmin(self.data) < self.norm.vmin:
            extend = "min"

        return extend

    def plot_colormap(
        self,
        method="pcolorfast",
        cmap="viridis",
        vmin=None,
        vmax=None,
        interval=None,
        stretch=None,
    ):
        if stretch is None:
            stretch = LinearStretch()
        self.norm = self._normalize(
            vmin=vmin, vmax=vmax, interval=interval, stretch=stretch
        )

        for i, v in enumerate(self.v):
            ax = self.imgrid[i]
            data = self.data[i]

            plot = getattr(self, "_" + method)
            im = plot(ax, data, cmap=cmap, norm=self.norm)

            ax.annotate(
                text="${:.2f}$ km s$^{{-1}}$".format(v),
                xy=(0.97, 0.95),
                ha="right",
                va="top",
                xycoords="axes fraction",
                color="black",
                fontsize=8,
                path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()],
            )

        ax.invert_xaxis()

        if self.colorbar:
            for ax in self.imgrid.cbar_axes:
                cbar = ax.colorbar(im, extend=self._set_colorbar_extend())
                cbar.set_label(self.cbar_label, rotation=270, labelpad=15)

        return

    def overlay_contour(
        self,
        fitsname_or_data="self",
        data_scaling_factor=1.0,
        levels=5,
        color="grey",
        linewidth=1.0,
        linestyle="solid",
        **kwargs
    ):

        if fitsname_or_data == "self":
            x = self.x
            y = self.y
            v = self.v
            data = self.data
        elif isinstance(fitsname_or_data, str):
            image = FitsImage(
                fitsname=fitsname_or_data,
                squeeze=True,
                rel_dir_ax=True,
                xlim=self.xlim,
                ylim=self.ylim,
                vlim=self.vlim,
            )

            if self.center_coord is not None:
                image.shift_phasecenter_toward(self.center_coord)
            x = image.x
            y = image.y
            if hasattr(image, "v"):
                v = image.v
            else:
                v = self.v
            data = image.data * data_scaling_factor
        else:
            data = fitsname_or_data * data_scaling_factor
            x = self.x
            y = self.y
            v = self.v
        
        if np.all(np.diff(v) < 0):
            v = v[::-1]
            if data.ndim == 3:
                data = data[::-1]

        for i, v_self in enumerate(self.v):
            ax = self.imgrid[i]
            if data.ndim == 3:
                d = data[np.argmin(abs(v - v_self))]
            else:
                d = data

            self.contour = self._contour(
                ax,
                x,
                y,
                d,
                levels=levels,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                **kwargs
            )

    def _pcolorfast(self, ax, data, cmap="viridis", norm=None):
        im = ax.pcolorfast(self.x, self.y, data, rasterized=True, cmap=cmap, norm=norm)
        return im

    def _normalize(self, vmin=None, vmax=None, interval=None, stretch=LinearStretch()):
        norm = ImageNormalize(
            self.data, vmin=vmin, vmax=vmax, interval=interval, stretch=stretch
        )
        return norm

    def _contour(
        self, ax, x, y, data, levels=5, color="black", linewidth=1.0, linestyle="solid", **kwargs
    ):
        im = ax.contour(
            x,
            y,
            data,
            levels=levels,
            colors=color,
            linewidths=linewidth,
            linestyles=linestyle,
            **kwargs
        )
        return im

    ### ANIMATION

    def animate(self, fig, ax):

        ims = [ig.get_images().copy() for ig in self.imgrid]

        for im in ims:
            ax.add_image(im[0])

        import matplotlib.animation as animation

        ani = animation.ArtistAnimation(fig, ims, interval=100)

        return ani

    ### ADDENDA

    def add_beam(self, mode="1", loc="lower left", color="white", fill=True, hatch="///////////"):
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse

        width = self.bmaj
        height = self.bmin
        angle = 90 - self.bpa  # to be measured from east

        if mode == "1":
            axes = [self.imgrid.axes_llc]
        elif mode == "all":
            axes = self.imgrid.axes_all

        for ax in axes:
            beam = AnchoredEllipse(
                ax.transData,
                width=width,
                height=height,
                angle=angle,
                loc=loc,
                pad=0.5,
                borderpad=0.5,
                frameon=False,
            )
            beam.ellipse.set(color=color, fill=fill, hatch=hatch)
            ax.add_artist(beam)

    ### APPEARANCE

    def set_title(self, title):
        self.fig.suptitle(title, y=0.9)

    def _set_xlabel(self, ax, xlabel):
        ax.set_xlabel(xlabel)

    def _set_ylabel(self, ax, ylabel):
        ax.set_ylabel(ylabel)

    def set_labels(self, mode="1", xlabel=None, ylabel=None):
        self.imgrid.set_label_mode(mode)

        for ax in self.imgrid.axes_all:
            self._set_xlabel(ax, xlabel)
            self._set_ylabel(ax, ylabel)

    def _get_major_ticker_locator(self, nticks=None, interval=None):
        if nticks is not None:
            return ticker.MaxNLocator(nticks)
        if interval is not None:
            return ticker.MultipleLocator(interval)
        return ticker.AutoLocator()

    def _get_minor_ticker_locator(self, nticks=None, interval=None):
        if nticks is not None:
            return ticker.MaxNLocator(nticks)
        if interval is not None:
            return ticker.MultipleLocator(interval)
        return ticker.AutoMinorLocator()

    def set_xaxis_ticker(
        self,
        major=True,
        minor=False,
        majornticks=None,
        minornticks=None,
        majorinterval=None,
        minorinterval=None,
    ):
        if not major:
            majorlocator = ticker.NullLocator()
        else:
            majorlocator = self._get_major_ticker_locator(
                nticks=majornticks, interval=majorinterval
            )

        if not minor:
            minorlocator = ticker.NullLocator()
        else:
            minorlocator = self._get_minor_ticker_locator(
                nticks=minornticks, interval=minorinterval
            )

        self.imgrid.axes_llc.xaxis.set_major_locator(majorlocator)
        self.imgrid.axes_llc.xaxis.set_minor_locator(minorlocator)

    def set_yaxis_ticker(
        self,
        major=True,
        minor=False,
        majornticks=None,
        minornticks=None,
        majorinterval=None,
        minorinterval=None,
    ):
        if not major:
            majorlocator = ticker.NullLocator()
        else:
            majorlocator = self._get_major_ticker_locator(
                nticks=majornticks, interval=majorinterval
            )

        if not minor:
            minorlocator = ticker.NullLocator()
        else:
            minorlocator = self._get_minor_ticker_locator(
                nticks=minornticks, interval=minorinterval
            )

        self.imgrid.axes_llc.yaxis.set_major_locator(majorlocator)
        self.imgrid.axes_llc.yaxis.set_minor_locator(minorlocator)

    def set_ticker(
        self,
        major=True,
        minor=False,
        majornticks=None,
        minornticks=None,
        majorinterval=None,
        minorinterval=None,
    ):
        self.set_xaxis_ticker(
            major=major,
            minor=minor,
            majornticks=majornticks,
            minornticks=minornticks,
            majorinterval=majorinterval,
            minorinterval=minorinterval,
        )
        self.set_yaxis_ticker(
            major=major,
            minor=minor,
            majornticks=majornticks,
            minornticks=minornticks,
            majorinterval=majorinterval,
            minorinterval=minorinterval,
        )

        if minor:
            self.imgrid.axes_llc.minorticks_on()


# class Spectrum(FitsImage):


#     def __init__(self, fitsname_or_data, x=None, yerr=None, squeeze=True, rel_dir_ax=True, xlim=None, ylim=None, vlim=None, downsample=False):
#         if isinstance(fitsname_or_data, str):
#             super().__init__(fitsname_or_data, squeeze=squeeze, rel_dir_ax=rel_dir_ax, xlim=xlim, ylim=ylim, vlim=vlim, downsample=downsample)

#         else:
#             self.avgspec = fitsname_or_data
#             self.x =


def get_figsize(ncols, nrows, max_size=()):
    return (ncols * 3, nrows * 3)


def get_imagegrid(npanel, pad=0.0, colorbar=True):
    ncols = np.ceil(npanel**0.5).astype(int)
    nrows = np.ceil(npanel / ncols).astype(int)
    fig = plt.figure(figsize=get_figsize(ncols, nrows))

    cbar_mode = "single" if colorbar else None

    imgrid = ImageGrid(
        fig,
        rect=111,
        nrows_ncols=(nrows, ncols),
        share_all=True,
        axes_pad=pad,
        cbar_mode=cbar_mode,
    )
    return fig, imgrid


# def left_bottom_ax(imgrid):
#     nrows, ncols = imgrid.get_geometry()
#     i = ncols * (nrows - 1)
#     return imgrid[i]


def plot_channel_map(
    fitsname,
    center_coord=None,
    rmax=10.0,
    vrange=None,
    thin=1,
    sigma_clip=None,
    rms=None,
    noisemask_kw=dict(rmin=10.0, rmax=15.0),
    pad=0.0,
    colorbar=True,
    cmap_kw=dict(),
    beam_kw=dict(),
):
    # load the imagecube
    print("Loading data...")
    imagecube = FitsImage(fitsname)
    imagecube.get_directional_coord(center_coord=center_coord)
    imagecube.get_spectral_coord()

    # measure the rms for better visualization
    if rms is None and sigma_clip is not None:
        print("Estimating rms...")
        imagecube.estimate_rms(**noisemask_kw)
        rms = imagecube.rms
        print("rms: {:.2f} mJy/beam".format(imagecube.rms * 1e3))

    # select the velocity channels to plot
    velax = imagecube.v
    if vrange is not None:
        velax = velax[is_within(imagecube.v, vrange)]
    velax = velax[
        ::thin
    ]  # skip each *thin* channel to reduce the number of channels to plot

    # setup imagegrid
    fig, imgrid = get_imagegrid(velax.size, pad=pad, colorbar=colorbar)

    # image normalization and clipping
    data = imagecube.data
    if vrange is not None:
        data = data[is_within(imagecube.v, vrange), :, :]  # limit to relevant channels
    data = data[
        ::thin
    ]  # skip each *thin* channel to reduce the number of channels to plot

    if sigma_clip is not None:
        data[data < sigma_clip * rms] = np.nan
    norm = ImageNormalize(
        data, vmin=sigma_clip * rms if sigma_clip is not None else 0.0
    )

    cmap_kw["norm"] = cmap_kw.get("norm", norm)

    # iterate over channels to plot
    for i, v in enumerate(velax):
        # define ax
        ax = imgrid[i]

        # get data
        # im = data[imagecube.v == v, :, :].squeeze()
        im = data[i]

        # plot
        print("Plotting v = {:.2f} km/s...".format(v))
        im = plot_2D_map(
            data=im,
            X=imagecube.x,
            Y=imagecube.y,
            ax=ax,
            cmap=True,
            cmap_method="pcolorfast",
            contour=False,
            colorbar=False,
            cmap_kw=cmap_kw,
        )

        # set the spatial range
        ax.set(xlim=(rmax, -rmax), ylim=(-rmax, rmax))

        # annotate the velocity value of the channel
        ax.annotate(
            text="{:.2f} km/s".format(v),
            xy=(0.95, 0.95),
            ha="right",
            va="top",
            xycoords="axes fraction",
            color="black",
            path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()],
        )

        # set ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    # colrobar
    imgrid.cbar_axes[0].colorbar(im)

    ### ImageGrid should have param of *ngrids*, but specifying this param cause an error (maybe bug?).
    ### Here is workaround for that, removing axes on which no data are drawn.
    for i in range(i + 1, len(imgrid)):
        imgrid[i].set_axis_off()

    # axes label in the bottom left panel
    imgrid.axes_llc.set(xlabel="$\Delta$R.A. [arcsec]", ylabel="$\Delta$Dec. [arcsec]")

    # add beam
    add_beam(ax=imgrid.axes_llc, beam=imagecube.beam, **beam_kw)

    return fig
