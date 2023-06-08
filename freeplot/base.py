

from typing import Iterable, Tuple, Optional, Dict, Union
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches

from .unit import UnitPlot
from .utils import style_env

 

class FreePlot(UnitPlot):


    @style_env
    def barplot(
        self, x: str, y: str, 
        data: pd.DataFrame, hue: Optional[str] = None, 
        index: Union[Tuple[int], str] = (0, 0), 
        orient: str = 'v',
        auto_fmt: bool = False, *,
        hatch: Optional[Iterable] = None,
        hatch_scale: int = 3,
        errorbar: str = 'sd',
        capsize: float = 0.1,
        style: Union[str, Iterable[str]] = 'bar',
        **kwargs
    ) -> None:
        """ Bar plotting according to pd.DataFrame. 
        See [here](https://seaborn.pydata.org/generated/seaborn.barplot.html?highlight=barplot) for details.

        Parameters:
        -----------
        x, y, hue: The colnames of x, y and hue.
        data: Dataset includes x, y, and hue.
        orient: 'v' or 'h'
            Plot the bar vertically (`v`) or horizontally (`h`).
        auto_fmt: `True`: Adjust the xticklabel.

        hatch: ["", "/", "//”, "//\\\\", "x", "+", ".", "*"]
        hatch_scale: hatch * hatch_scale.
        kwargs: other kwargs for sns.barplot
            - palette: Dict|List, 
                set the color for each of hue.
            - edgecolor: str
                the edgecolor of the bar
            - width: float, the width of a full element when not using hue nesting, 
                or width of all the elements for one level of the major grouping variable.
            - ci: float or 'sd', optional, `sd`: Skip bootstrapping and draw the standard deviation of the observations.  
                `None`: No bootstrapping will be performed, and error bars will not be drawn.
            - errorbar: str, name of errorbar method (either “ci”, “pi”, “se”, or “sd”), 
                    or a tuple with a method name and a level parameter, 
                    or a function that maps from a vector to a (min, max) interval.
            - ...
        
        Examples:
        ---------
        >>> A = [1., 2., 3.]
        >>> B = [2., 3., 4.]
        >>> T = ['One', 'Two', 'Three'] * 2
        >>> Hue = ['A'] * len(A) + ['B'] * len(B)
        >>> data = pd.DataFrame(
        ...    {
        ...        "T": T,
        ...        "val": A + B,
        ...        "category": Hue
        ...    }
        ... )
        >>> fp = FreePlot(dpi=300)
        >>> fp.barplot(x='T', y='val', hue='category', data=data, index=(0, 0), auto_fmt=True)
        # using hatch
        >>> fp.barplot(x='T', y='val', hue='category', data=data, palette=['white'], edgecolor='black', hatch=['', '/', '//'])

        """
        ax = self[index]
        sns.barplot(
            x=x, y=y, hue=hue, data=data, ax=ax, orient=orient,
            errorbar = errorbar, capsize=capsize,
            **kwargs
        )
        if auto_fmt:
            self.fig.autofmt_xdate()
        if hatch:
            for pattern, bars in zip(hatch, self.get_container(index=index)):
                for bar in bars:
                    bar.set_hatch(pattern * hatch_scale)
        return self.get_legend_handles_labels(index)

    @style_env
    def contourf(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
        levels: Optional[Union[int, np.ndarray]] = 5, cbar: bool = True,
        index: Union[Tuple[int], str] = (0, 0), *,
        style: Union[str, Iterable[str]] = [],
        origin: Optional[str] = 'lower', cmap = plt.cm.bone,
        **kwargs
    ):
        """Plot filled contours.
        See [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.contourf.html?highlight=contourf#matplotlib.axes.Axes.contourf) for details.

        Parameters:
        -----------
        X, Y: The coordinates of the values in Z.
        Z: (M, N), the height values over which the contour is draw.
        levels: Determines the number and positions of the contour lines / regions.
        cbar: `True`: Add color bar.
        origin: {None, 'upper', 'lower', 'image'}. 
            Determines the orientation and exact position of Z by specifying the position of Z[0, 0].
        cmap: The Colormap instance or registered colormap name used to map scalar data to colors.
        kwargs: other kwargs for `contourf`
            - linewidths: float or array-like
            - linestyles: {None, 'solid', 'dashed', 'dashdot', 'dotted'}
            - hatches: list[str]
        
        Examples:
        ---------
        >>> X = np.arange(-5, 5, 0.25)
        >>> Y = np.arange(-5, 5, 0.25)
        >>> X, Y = np.meshgrid(X, Y)
        >>> R = np.sqrt(X**2 + Y**2)
        >>> Z = np.sin(R)
        >>> fp = FreePlot(dpi=300)
        >>> fp.contourf(X, Y, Z, levels=5, cmap=plt.cm.bone)

        """
        ax = self[index]
        cs =  ax.contourf(X, Y, Z, levels, cmap=cmap, origin=origin, **kwargs)
        if cbar:
            self.fig.colorbar(cs)
        return cs

    @style_env
    def histplot(
        self, x: np.ndarray, 
        num_bins: int, density: bool = False, range: Optional[Tuple] = None,
        cumulative: bool = False, histtype: str = 'bar',
        index: Union[Tuple[int], str] = (0, 0), *,
        style: Union[str, Iterable[str]] = 'hist',
        **kwargs
    ):
        r"""
        Compute and plot a histogram.
        See [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html#matplotlib.axes.Axes.hist) for more details.

        Parameters:
        -----------
        x: (n,) array
        num_bins: int
            The number bins.
        density: bool, default to False
            `True`: the counts are normalized to `1`
        cumulative: bool, default to False
            `True`: a histogram is computed where each bin gives the counts in that bin plus all bins for smaller values
        range: tuple, optional
            `None`: range will be (x.min(), x.max())

        Returns:
        --------
        n: array
            The number of bins.
        bins: array
            The edges of the bins with a length of `n + 1`.
        patches: BarContainer

        Examples:
        ---------
        >>> x = np.random.rand(1024)
        >>> fp.histplot(x, num_bins=100, density=True)
        """
        ax = self[index]
        ax.hist(
            x, bins=num_bins, 
            density=density, range=range,
            cumulative=cumulative, histtype=histtype,
            **kwargs
        )


    @style_env
    def heatmap(
        self, data: pd.DataFrame, 
        index: Union[Tuple[int], str] = (0, 0), 
        annot: bool = True, 
        fmt: str = ".4f",
        cmap: str = 'GnBu', 
        linewidth: float = .5, *,
        style: Union[str, Iterable[str]] = 'heatmap',
        **kwargs
    ) -> None:
        """Plot rectangular data as a color-encoded matrix.
        See https://seaborn.pydata.org/generated/seaborn.heatmap.html?highlight=heatmap#seaborn.heatmap for details.

        Parameters:
        -----------
        data: (M, N), Dataset.
        cmap: colormap, GnBu, Oranges are recommanded.
        annot: Annotation.
        fmt: the format of annotation.
        **kwargs: other kwargs for `sns.heatmap`
            - cbar: bool
                `True`: Add color bar.

        Examples:
        ---------
        >>> titles = ("S", "h", "a", "n")
        >>> row_labels = ('c', 'u', 't', 'e')
        >>> col_labels = ('l', 'r', 'i', 'g')
        >>> data = np.random.rand(4, 4)
        >>> df = pd.DataFrame(data, index=col_labels, columns=row_labels)
        >>> fp = FreePlot()
        >>> fp.heatmap(df, annot=True, fmt=".4f", cbar=False, linewidth=0.5)

        """
        ax = self[index]
        return sns.heatmap(
            data, ax=ax, 
            annot=annot, fmt=fmt,
            cmap=cmap, linewidth=linewidth,
            **kwargs
        )


    @style_env
    def imageplot(
        self, img: np.ndarray, 
        index: Union[Tuple[int], str] = (0, 0), 
        show_ticks: bool = False, *, 
        style: Union[str, Iterable[str]] = 'image',
        **kwargs
    ) -> None:
        """Display data as an image, i.e., on a 2D regular raster.
        See [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html?highlight=imshow#matplotlib.pyplot.imshow) for details.

        Parameters:
        -----------
        img: Image.
        show_ticks: bool
            - `True`: Show the ticks.
        **kwargs: other kwargs for `ax.imshow`
            - cmap: str or colormap
            - norm: Normalization method.
            - vmin, vmax: float
            - ...

        Examples:
        ---------
        >>> fp.imageplot(img, show_ticks=False)

        """
        ax = self[index]
        img = img[..., None]
        try:
            assert img.shape[2] == 3
            ax.imshow(img.squeeze(), **kwargs)
        except AssertionError:
            if not kwargs.get('cmap', False):
                kwargs['cmap'] = 'gray'
            ax.imshow(img.squeeze(), **kwargs)
        if not show_ticks:
            ax.axis('off')


    @style_env
    def lineplot(
        self, x: np.ndarray, y: np.ndarray, 
        index: Union[Tuple[int], str] = (0, 0), *, 
        style: Union[str, Iterable[str]] = 'line',
        **kwargs
    ) -> None:
        """Draw a line plot.
        See [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html?highlight=plot#matplotlib.axes.Axes.plot) for details.

        Parameters:
        -----------
        x, y: array-like 
            Coordinates.
        **kwargs: other kwargs for ax.plot()
            - marker: `''`: No markers.
            - ...
        
        Examples:
        ---------
        >>> x = np.linspace(-10, 10, 20)
        >>> y = np.sin(x) + np.random.randn(20)
        >>> fp = FreePlot((1, 2), (4.4, 2), dpi=300, sharey=True)
        >>> fp.lineplot(x, y, index=(0, 0), style='line')
        >>> # plotting a line without markers
        >>> fp.lineplot(x, y, index=(0, 1), marker='')

        """
        ax = self[index]
        return ax.plot(x, y, **kwargs)


    @style_env
    def stackplot(
        self, x: np.ndarray, y: np.ndarray, 
        index: Union[Tuple[int], str] = (0, 0), *, 
        style: Union[str, Iterable[str]] = 'stack',
        **kwargs
    ) -> None:
        """Draw a stacked area plot.
        See [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.stackplot.html#matplotlib.axes.Axes.stackplot) for details.

        Parameters:
        -----------
        x: (N,) array-like
        y: (M, N) array-like
        **kwargs: other kwargs for ax.stackplot()
            - labels: list of str
            - colors: list of color
            - All other keyword arguments are passed to Axes.fill_between
        
        Examples:
        ---------
        >>> x = np.arange(0, 10, 2)
        >>> ay = [1, 1.25, 2, 2.75, 3]
        >>> by = [1, 1, 1, 1, 1]
        >>> cy = [2, 1, 2, 1, 2]
        >>> y = np.vstack([ay, by, cy])
        >>> fp = FreePlot()
        >>> fp.stackplot(x, y, index=(0, 0), style='stack')

        """
        ax = self[index]
        return ax.stackplot(x, y, **kwargs)

        
    @style_env
    def scatterplot(
        self, x: np.ndarray, y: np.ndarray, 
        index: Union[Tuple[int], str] = (0, 0), *,
        style: Union[str, Iterable[str]] = 'scatter',
        **kwargs
    ) -> None:
        """A scatter plot of y vs. x with varying marker size and/or color.
        See [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html?highlight=scatter#matplotlib.axes.Axes.scatter) for details.

        Parameters:
        -----------
        x, y: array-like
            Coordinates.
        **kwargs: other kwargs for `ax.scatter`
            - s: float or array-like, the marker size
            - c: array-like or list of colors or color, the marker colors
            - marker: marker style
            - cmap: color map
            - vmin, vmax:
            - alpha:
            - linewidth:
            - edgecolors: {'face', 'none', None} or color or sequence of color
            - ...

        Examples:
        ---------
        >>> from scipy.stats import multivariate_normal
        >>> nums = 100
        >>> means = (
        ...    (0, 0),
        ...    (5, 5),
        ...    (-5, -5)
        >>> )
        >>> cov = 2
        >>> data = multivariate_normal.rvs(mean, cov, size=nums)
        >>> x, y = data[:, 0], data[:, 1]
        >>> fp.scatterplot(x, y, edgecolors='none')

        """
        ax = self[index]
        return ax.scatter(x, y, **kwargs)

    def surfaceplot(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
        index: Union[Tuple[int], str] = (0, 0), *,
        style: Union[str, Iterable[str]] = 'surface',
        cmap = plt.cm.coolwarm, antialiased=False,
        **kwargs
    ):
        """Create a surface plot.
        See [here](https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html?highlight=plot_surface#mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface) for details.

        Parameters:
        -----------
        X, Y, Z: 2D arrary.
        cmap: Colormap.
        antialiased: 
        **kwargs: other kwargs of `ax.plot_surface`
            - color: color-like 
                Color of the surface patches.
            - facecolors: array-like of colors
                Colors of each individual patch.
            - ...

        Examples:
        ---------
        >>> X = np.arange(-5, 5, 0.25)
        >>> Y = np.arange(-5, 5, 0.25)
        >>> X, Y = np.meshgrid(X, Y)
        >>> R = np.sqrt(X**2 + Y**2)
        >>> Z = np.sin(R)
        >>> fp = FreePlot(projection='3d', dpi=300)
        >>> fp.surfaceplot(X, Y, Z, cmap=plt.cm.coolwarm, antialiased=False, linewidth=0)

        """
        ax = self[index]
        results = ax.plot_surface(X, Y, Z, cmap=cmap, antialiased=antialiased, **kwargs)
        ax.tick_params('x', pad=0.01)
        ax.tick_params('y', pad=0.01)
        ax.tick_params('z', pad=0.01)
        return results

    @style_env
    def violinplot(
        self, y: Iterable, x: Optional[Iterable[str]] = None,
        index: Union[Tuple[int], str] = (0, 0), *,
        style: Union[str, Iterable[str]] = 'violin',
        **kwargs
    ) -> None:
        """Make a violin plot.
        See [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.violinplot.html?highlight=violinplot#matplotlib.axes.Axes.violinplot) for details.

        Parameters:
        -----------
        y: Dataset, each of y is a group of data.
        x: Group index.
        **kwargs: other kwargs for `ax.violinplot`
            - positions: array-like
                The positions of the violins. The ticks and limits are automatically set to match the positions.
            - vert: bool 
                `True`: creates a vertical violin plot. Otherwise, creates a horizontal violin plot.
            - widths: 
            - showmeans: bool, default: False
            - showextrema: bool, default: True
            - showmedians: bool, default: False

        Examples:
        ---------
        >>> # note that each element is a group of data ...
        >>> dataset = [np.random.normal(0, std, 100) for std in range(5, 10)]
        >>> fp.violinplot(x=None, y=dataset, index=(0, 0))
        >>> fp.violinplot(x=[f"std-{std}" for std in range(5, 10)], y=dataset, index=(0, 0))

        """

        if x is None:
            x = range(1, len(y) + 1)
        ax = self[index]
        obj = ax.violinplot(dataset=y, **kwargs)
        ax.set(
            xticks=range(1, len(y) + 1),
            xticklabels=x
        )
        for key in ['cmaxes', 'cmins', 'cbars']:
            try:
                obj[key].set_linewidth(0.1)
            except KeyError:
                pass
        return obj


    def add_patch(self, patch: patches.Patch, index: Union[Tuple[int], str] = (0, 0)) -> patches.Patch:
        """Add patch to the Axes."""
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

    def __init__(
        self, alpha: int = 1., fill: bool = False,
        linewidth: float = None, linestyle: str = None, hatch: str = None,
        capstyle: str = None, joinstyle: str = None
    ) -> None:
        self.__share = {
            'alpha': alpha,
            'fill': fill,
            'linewidth': linewidth,
            'linestyle': linestyle,
            'hatch': hatch,
            'capstyle': capstyle,
            'joinstyle': joinstyle
        }

    @property
    def share(self):
        return self.__share.copy()

    @_redirect(patches)
    def Annulus(self, x:float, y:float, width:float, angle: float = 0., **kwargs):
        return ((x, y), width, angle), kwargs

    @_redirect(patches, ['fill'])
    def Arc(
        self, x:float, y:float, width:float, height:float, angle: float = 0.,
        theta1: float = 0., theta2: float = 0., **kwargs
    ):
        return ((x, y), width, height, angle, theta1, theta2), kwargs

    @_redirect(patches)
    def Arrow(self, x:float, y:float, dx:float, dy:float, **kwargs):
        return (x, y, dx, dy), kwargs

    @_redirect(patches)
    def Circle(self, x:float, y:float, radius: float, **kwargs):
        return ((x, y), radius), kwargs

    @_redirect(patches)
    def CirclePolygon(self, x:float, y:float, resolution: float = 20, **kwargs):
        return ((x, y), resolution), kwargs

    @_redirect(patches)
    def ConnectionPatch(self, *args, **kwargs):
        return args, kwargs

    @_redirect(patches)
    def Ellipse(self, x:float, y:float, width:float, height:float, angle:float = 0., **kwargs):
        return ((x, y), width, height, angle), kwargs

    @_redirect(patches)
    def Polygen(self, x:np.ndarray, y:np.ndarray, closed: bool = True, **kwargs):
        assert x.ndim == y.ndim == 1, "check: x.ndim == y.ndim == 1"
        xy = np.vstack((x, y)).T
        return (xy, closed), kwargs

    @_redirect(patches)
    def Rectangle(self, x:float, y:float, width:float, height:float, angle: float = 0., **kwargs):
        """
        (x, y) represents the left bottom corner for the common cartesian coordinate system
        while the left upper corner in the case of image plotting.
        """
        return ((x, y), width, height, angle), kwargs








