from typing import Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from matplotlib.container import Container

from .unit import UnitPlot
from .utils import style_env


class FreePlot(UnitPlot):
    r"""High-level plotting container.

    `FreePlot` extends `UnitPlot` with common chart helpers while preserving
    direct access to Matplotlib axes.
    """

    @style_env
    def barplot(
        self,
        x: str,
        y: str,
        data: pd.DataFrame,
        hue: Optional[str] = None,
        index: Union[Tuple[int], str] = (0, 0),
        orient: str = "v",
        auto_fmt: bool = False,
        *,
        hatch: Optional[Iterable] = None,
        hatch_scale: int = 3,
        errorbar: str = "sd",
        capsize: float = 0.1,
        style: Union[str, Iterable[str]] = "bar",
        **kwargs,
    ) -> None:
        r"""Bar plotting according to pd.DataFrame.

        Parameters
        ----------
        x : str
            Column name for the x-axis variable.
        y : str
            Column name for the y-axis variable.
        data : pd.DataFrame
            Data source containing `x`, `y`, and optional `hue` columns.
        hue : str, optional
            Column name used to split bars into groups.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        orient : {"v", "h"}, default="v"
            Bar orientation.
        auto_fmt : bool, default=False
            Whether to auto-format x tick labels.
        hatch : iterable, optional
            Hatch patterns applied to bar containers.
        hatch_scale : int, default=3
            Repetition count for each hatch pattern.
        errorbar : str, default="sd"
            Seaborn error bar method.
        capsize : float, default=0.1
            Width of error bar caps.
        style : str or iterable of str, default="bar"
            Style names resolved by `style_env`.
        **kwargs
            Additional keyword arguments passed to `seaborn.barplot`.

        Returns
        -------
        tuple
            Legend handles and labels from the target axes.

        Examples
        --------
        >>> data = pd.DataFrame({"name": ["A", "B"], "value": [1.0, 2.0]})
        >>> fp = FreePlot()
        >>> fp.barplot(x="name", y="value", data=data)
        """
        ax = self[index]
        sns.barplot(
            x=x,
            y=y,
            hue=hue,
            data=data,
            ax=ax,
            orient=orient,
            errorbar=errorbar,
            capsize=capsize,
            **kwargs,
        )
        if auto_fmt:
            self.fig.autofmt_xdate()
        if hatch:
            for pattern, bars in zip(hatch, self.get_containers(index=index)):
                for bar in bars:
                    bar.set_hatch(pattern * hatch_scale)
        if hatch and hue:  # hatched legend
            handles, labels = ax.get_legend_handles_labels()
            for h, pattern in zip(handles, hatch):
                if isinstance(h, Container):
                    for bar in h:
                        bar.set_hatch(pattern * hatch_scale)
                else:
                    h.set_hatch(pattern * hatch_scale)
            ax.legend(handles=handles, labels=labels)
        return self.get_legend_handles_labels(index)

    @style_env
    def contourf(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        levels: Optional[Union[int, np.ndarray]] = 5,
        cbar: bool = True,
        index: Union[Tuple[int], str] = (0, 0),
        *,
        style: Union[str, Iterable[str]] = [],
        origin: Optional[str] = "lower",
        cmap=plt.cm.bone,
        **kwargs,
    ):
        r"""Plot filled contours.

        Parameters
        ----------
        X : np.ndarray
            X coordinates.
        Y : np.ndarray
            Y coordinates.
        Z : np.ndarray
            Height values over which contours are drawn.
        levels : int or np.ndarray, optional
            Number or positions of contour levels.
        cbar : bool, default=True
            Whether to add a color bar.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        style : str or iterable of str, default=[]
            Style names resolved by `style_env`.
        origin : str, optional
            Orientation and position of `Z[0, 0]`.
        cmap : colormap, optional
            Colormap used to map scalar values to colors.
        **kwargs
            Additional keyword arguments passed to `Axes.contourf`.

        Returns
        -------
        matplotlib.contour.QuadContourSet
            Created contour set.

        Examples
        --------
        >>> X = np.arange(-5, 5, 0.25)
        >>> Y = np.arange(-5, 5, 0.25)
        >>> X, Y = np.meshgrid(X, Y)
        >>> Z = np.sin(np.sqrt(X**2 + Y**2))
        >>> fp = FreePlot()
        >>> fp.contourf(X, Y, Z, levels=5)
        """
        ax = self[index]
        cs = ax.contourf(X, Y, Z, levels, cmap=cmap, origin=origin, **kwargs)
        if cbar:
            self.fig.colorbar(cs)
        return cs

    @style_env
    def histplot(
        self,
        x: np.ndarray,
        num_bins: int,
        density: bool = False,
        range: Optional[Tuple] = None,
        cumulative: bool = False,
        histtype: str = "bar",
        color: str = "#0050C2",
        index: Union[Tuple[int], str] = (0, 0),
        *,
        style: Union[str, Iterable[str]] = "hist",
        **kwargs,
    ):
        r"""Compute and plot a histogram.

        Parameters
        ----------
        x : np.ndarray
            Input values.
        num_bins : int
            Number of bins.
        density : bool, default=False
            Whether to normalize counts to form a density.
        range : tuple, optional
            Lower and upper range of bins.
        cumulative : bool, default=False
            Whether each bin includes counts from previous bins.
        histtype : str, default="bar"
            Histogram type passed to `Axes.hist`.
        color : str, default="#0050C2"
            Histogram color.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        style : str or iterable of str, default="hist"
            Style names resolved by `style_env`.
        **kwargs
            Additional keyword arguments passed to `Axes.hist`.

        Examples
        --------
        >>> x = np.random.rand(1024)
        >>> fp.histplot(x, num_bins=100, density=True)
        """
        ax = self[index]
        ax.hist(
            x,
            bins=num_bins,
            density=density,
            range=range,
            color=color,
            cumulative=cumulative,
            histtype=histtype,
            **kwargs,
        )

    @style_env
    def heatmap(
        self,
        data: pd.DataFrame,
        index: Union[Tuple[int], str] = (0, 0),
        annot: bool = True,
        fmt: str = ".4f",
        cmap: str = "GnBu",
        linewidth: float = 0.5,
        *,
        style: Union[str, Iterable[str]] = "heatmap",
        **kwargs,
    ) -> None:
        r"""Plot rectangular data as a color-encoded matrix.

        Parameters
        ----------
        data : pd.DataFrame
            Matrix-like data to plot.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        annot : bool, default=True
            Whether to write data values in cells.
        fmt : str, default=".4f"
            Annotation format string.
        cmap : str, default="GnBu"
            Colormap name.
        linewidth : float, default=0.5
            Width of lines that divide cells.
        style : str or iterable of str, default="heatmap"
            Style names resolved by `style_env`.
        **kwargs
            Additional keyword arguments passed to `seaborn.heatmap`.

        Returns
        -------
        matplotlib.axes.Axes
            Target axes.

        Examples
        --------
        >>> df = pd.DataFrame(np.random.rand(4, 4))
        >>> fp = FreePlot()
        >>> fp.heatmap(df, annot=True, cbar=False)
        """
        ax = self[index]
        return sns.heatmap(
            data, ax=ax, annot=annot, fmt=fmt, cmap=cmap, linewidth=linewidth, **kwargs
        )

    @style_env
    def imageplot(
        self,
        img: np.ndarray,
        index: Union[Tuple[int], str] = (0, 0),
        show_ticks: bool = False,
        *,
        style: Union[str, Iterable[str]] = "image",
        **kwargs,
    ) -> None:
        r"""Display data as an image.

        Parameters
        ----------
        img : np.ndarray
            Image data.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        show_ticks : bool, default=False
            Whether to keep axis ticks visible.
        style : str or iterable of str, default="image"
            Style names resolved by `style_env`.
        **kwargs
            Additional keyword arguments passed to `Axes.imshow`.

        Examples
        --------
        >>> fp.imageplot(img, show_ticks=False)
        """
        ax = self[index]
        img = img[..., None]
        try:
            assert img.shape[2] == 3
            ax.imshow(img.squeeze(), **kwargs)
        except AssertionError:
            if not kwargs.get("cmap", False):
                kwargs["cmap"] = "gray"
            ax.imshow(img.squeeze(), **kwargs)
        if not show_ticks:
            ax.axis("off")

    @style_env
    def lineplot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        index: Union[Tuple[int], str] = (0, 0),
        *,
        style: Union[str, Iterable[str]] = "line",
        **kwargs,
    ) -> None:
        r"""Draw a line plot.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        style : str or iterable of str, default="line"
            Style names resolved by `style_env`.
        **kwargs
            Additional keyword arguments passed to `Axes.plot`.

        Returns
        -------
        list[matplotlib.lines.Line2D]
            Created line artists.

        Examples
        --------
        >>> x = np.linspace(-10, 10, 20)
        >>> fp = FreePlot()
        >>> fp.lineplot(x, np.sin(x), marker="")
        """
        ax = self[index]
        return ax.plot(x, y, **kwargs)

    @style_env
    def stackplot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        index: Union[Tuple[int], str] = (0, 0),
        *,
        style: Union[str, Iterable[str]] = "stack",
        **kwargs,
    ) -> None:
        r"""Draw a stacked area plot.

        Parameters
        ----------
        x : np.ndarray
            X coordinates with shape `(N,)`.
        y : np.ndarray
            Stacked values with shape `(M, N)`.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        style : str or iterable of str, default="stack"
            Style names resolved by `style_env`.
        **kwargs
            Additional keyword arguments passed to `Axes.stackplot`.

        Returns
        -------
        list[matplotlib.collections.PolyCollection]
            Created stacked area artists.

        Examples
        --------
        >>> x = np.arange(0, 10, 2)
        >>> y = np.vstack([np.ones(5), np.arange(1, 6)])
        >>> fp = FreePlot()
        >>> fp.stackplot(x, y)
        """
        ax = self[index]
        return ax.stackplot(x, y, **kwargs)

    @style_env
    def scatterplot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        index: Union[Tuple[int], str] = (0, 0),
        *,
        style: Union[str, Iterable[str]] = "scatter",
        **kwargs,
    ) -> None:
        r"""A scatter plot of y vs. x with varying marker size and/or color.

        Parameters
        ----------
        x : np.ndarray
            X coordinates.
        y : np.ndarray
            Y coordinates.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        style : str or iterable of str, default="scatter"
            Style names resolved by `style_env`.
        **kwargs
            Additional keyword arguments passed to `Axes.scatter`.

        Returns
        -------
        matplotlib.collections.PathCollection
            Created scatter artist.

        Examples
        --------
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> fp = FreePlot()
        >>> fp.scatterplot(x, y, edgecolors="none")
        """
        ax = self[index]
        return ax.scatter(x, y, **kwargs)

    def surfaceplot(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        index: Union[Tuple[int], str] = (0, 0),
        *,
        style: Union[str, Iterable[str]] = "surface",
        cmap=plt.cm.coolwarm,
        antialiased=False,
        **kwargs,
    ):
        r"""Create a surface plot.

        Parameters
        ----------
        X : np.ndarray
            X coordinates.
        Y : np.ndarray
            Y coordinates.
        Z : np.ndarray
            Surface values.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        style : str or iterable of str, default="surface"
            Style names reserved for consistency with other plot methods.
        cmap : colormap, optional
            Colormap used to map scalar values to colors.
        antialiased : bool, default=False
            Whether to draw antialiased surface edges.
        **kwargs
            Additional keyword arguments passed to `Axes3D.plot_surface`.

        Returns
        -------
        mpl_toolkits.mplot3d.art3d.Poly3DCollection
            Created surface artist.

        Examples
        --------
        >>> X = np.arange(-5, 5, 0.25)
        >>> Y = np.arange(-5, 5, 0.25)
        >>> X, Y = np.meshgrid(X, Y)
        >>> Z = np.sin(np.sqrt(X**2 + Y**2))
        >>> fp = FreePlot(projection="3d")
        >>> fp.surfaceplot(X, Y, Z, linewidth=0)
        """
        ax = self[index]
        results = ax.plot_surface(X, Y, Z, cmap=cmap, antialiased=antialiased, **kwargs)
        ax.tick_params("x", pad=0.01)
        ax.tick_params("y", pad=0.01)
        ax.tick_params("z", pad=0.01)
        return results

    @style_env
    def violinplot(
        self,
        y: Iterable,
        x: Optional[Iterable[str]] = None,
        index: Union[Tuple[int], str] = (0, 0),
        *,
        style: Union[str, Iterable[str]] = "violin",
        **kwargs,
    ) -> None:
        r"""Make a violin plot.

        Parameters
        ----------
        y : iterable
            Dataset groups.
        x : iterable of str, optional
            Labels for dataset groups.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        style : str or iterable of str, default="violin"
            Style names resolved by `style_env`.
        **kwargs
            Additional keyword arguments passed to `Axes.violinplot`.

        Returns
        -------
        dict
            Violin plot artist dictionary returned by Matplotlib.

        Examples
        --------
        >>> dataset = [np.random.normal(0, std, 100) for std in range(1, 4)]
        >>> fp = FreePlot()
        >>> fp.violinplot(y=dataset, x=["one", "two", "three"])
        """

        if x is None:
            x = range(1, len(y) + 1)
        ax = self[index]
        obj = ax.violinplot(dataset=y, **kwargs)
        ax.set(xticks=range(1, len(y) + 1), xticklabels=x)
        for key in ["cmaxes", "cmins", "cbars"]:
            try:
                obj[key].set_linewidth(0.1)
            except KeyError:
                pass
        return obj

    def add_patch(
        self, patch: patches.Patch, index: Union[Tuple[int], str] = (0, 0)
    ) -> patches.Patch:
        r"""Add a patch to an axes.

        Parameters
        ----------
        patch : matplotlib.patches.Patch
            Patch artist to add.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.

        Returns
        -------
        matplotlib.patches.Patch
            Added patch artist.
        """
        ax = self[index]
        return ax.add_patch(patch)


def _redirect(module, exclude_keys: Optional[Iterable] = None):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            share = self.share
            args, kwargs = func(self, *args, **kwargs)
            share.update(kwargs)
            if exclude_keys is not None:
                for key in exclude_keys:
                    del share[key]
            return getattr(module, func.__name__)(*args, **share)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


class FreePatches:
    r"""Factory for Matplotlib patch objects with shared defaults.

    Parameters
    ----------
    alpha : float, default=1.0
        Patch opacity.
    fill : bool, default=False
        Whether patches are filled.
    linewidth : float, optional
        Patch line width.
    linestyle : str, optional
        Patch line style.
    hatch : str, optional
        Patch hatch pattern.
    capstyle : str, optional
        Patch cap style.
    joinstyle : str, optional
        Patch join style.
    """

    def __init__(
        self,
        alpha: int = 1.0,
        fill: bool = False,
        linewidth: float = None,
        linestyle: str = None,
        hatch: str = None,
        capstyle: str = None,
        joinstyle: str = None,
    ) -> None:
        self.__share = {
            "alpha": alpha,
            "fill": fill,
            "linewidth": linewidth,
            "linestyle": linestyle,
            "hatch": hatch,
            "capstyle": capstyle,
            "joinstyle": joinstyle,
        }

    @property
    def share(self):
        return self.__share.copy()

    @_redirect(patches)
    def Annulus(self, x: float, y: float, width: float, angle: float = 0.0, **kwargs):
        return ((x, y), width, angle), kwargs

    @_redirect(patches, ["fill"])
    def Arc(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        angle: float = 0.0,
        theta1: float = 0.0,
        theta2: float = 0.0,
        **kwargs,
    ):
        return ((x, y), width, height, angle, theta1, theta2), kwargs

    @_redirect(patches)
    def Arrow(self, x: float, y: float, dx: float, dy: float, **kwargs):
        return (x, y, dx, dy), kwargs

    @_redirect(patches)
    def Circle(self, x: float, y: float, radius: float, **kwargs):
        return ((x, y), radius), kwargs

    @_redirect(patches)
    def CirclePolygon(self, x: float, y: float, resolution: float = 20, **kwargs):
        return ((x, y), resolution), kwargs

    @_redirect(patches)
    def ConnectionPatch(self, *args, **kwargs):
        return args, kwargs

    @_redirect(patches)
    def Ellipse(self, x: float, y: float, width: float, height: float, angle: float = 0.0, **kwargs):
        return ((x, y), width, height, angle), kwargs

    @_redirect(patches)
    def Polygon(self, x: np.ndarray, y: np.ndarray, closed: bool = True, **kwargs):
        assert x.ndim == y.ndim == 1, "check: x.ndim == y.ndim == 1"
        xy = np.vstack((x, y)).T
        kwargs["closed"] = closed
        return (xy,), kwargs

    @_redirect(patches)
    def Rectangle(
        self, x: float, y: float, width: float, height: float, angle: float = 0.0, **kwargs
    ):
        """
        (x, y) represents the left bottom corner for the common cartesian coordinate system
        while the left upper corner in the case of image plotting.
        """
        return ((x, y), width, height, angle), kwargs
