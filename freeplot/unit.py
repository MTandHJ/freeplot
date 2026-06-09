from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.axes._axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

from .config import COLORS, MARKERS, cfg, style_cfg
from .utils import get_style, inherit_from_matplotlib, style_env


class UnitAX:
    r"""Lazy axes holder.

    Parameters
    ----------
    axes : FreeAxes
        Parent axes collection.
    position : matplotlib.gridspec.GridSpec
        Grid position in the figure.
    anchor : UnitAX, optional
        Anchor axes used for shared y-axis creation.
    sharey : bool, default=True
        Whether to share y-axis with `anchor`.
    **kwargs
        Additional keyword arguments passed to `Figure.add_subplot`.
    """

    def __init__(
        self,
        axes: "FreeAxes",
        position: matplotlib.gridspec.GridSpec,
        anchor: Optional["UnitAX"] = None,
        sharey: bool = True,
        **kwargs,
    ):
        self.axes = axes
        self.position = position
        self.anchor = anchor
        self.sharey = sharey
        self.kwargs = kwargs
        self.__ax = None

    @property
    def ax(self):
        r"""Create and return the wrapped Matplotlib axes.

        Notes
        -----
        Axes are created lazily so style settings can be applied before the
        first plotting call.

        Returns
        -------
        matplotlib.axes.Axes
            Created or cached axes.
        """
        if self.__ax is None:
            if not self.sharey or self.anchor is None:
                self.__ax = self.axes.fig.add_subplot(self.position, **self.kwargs)
            else:
                self.__ax = self.axes.fig.add_subplot(
                    self.position, sharey=self.anchor.ax, **self.kwargs
                )
                plt.setp(self.__ax.get_yticklabels(), visible=False)
        return self.__ax


class FreeAxes:
    r"""Grid collection of lazily created axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure that owns the axes.
    shape : tuple[int, int]
        Grid shape as `(rows, cols)`.
    titles : iterable, optional
        Titles used to access axes by name.
    sharey : bool, default=True
        Whether non-anchor axes in each row share y-axis with the row anchor.
    projection : str, optional
        Matplotlib projection name.
    """

    def __init__(
        self,
        fig: matplotlib.figure.Figure,
        shape: Tuple[int, int],
        titles: Optional[Iterable] = None,
        sharey: bool = True,
        projection: Optional[str] = None,
    ):
        assert len(shape) == 2, "Only grid-like Axes (#rows, #cols) are supported"

        self.fig = fig
        self.axes = []
        if titles is not None:
            titles = np.array(titles).reshape(shape)

        grids = fig.add_gridspec(*shape)
        for i in range(shape[0]):
            self.axes.append([])
            anchor = UnitAX(self, grids[i, 0], anchor=None, sharey=False, projection=projection)
            self.axes[-1].append(anchor)
            for j in range(1, shape[1]):
                ax = UnitAX(self, grids[i, j], anchor=anchor, sharey=sharey, projection=projection)
                self.axes[-1].append(ax)

        self.axes = np.array(self.axes)
        self.links = self._get_links(titles)
        self.titles = np.array(list(self.links.keys()))

    def _get_links(self, titles: Optional[Iterable]) -> Dict:
        r"""Map titles to axes indices.

        Parameters
        ----------
        titles : iterable, optional
            Titles arranged according to `self.axes.shape`.

        Returns
        -------
        dict
            Mapping from title to axes index.
        """
        m, n = self.axes.shape
        names = dict()
        if titles is None:
            for i in range(m):
                for j in range(n):
                    s = "(" + chr(i * n + j + 97) + ")"
                    names.update({s: (i, j)})
        else:
            for i in range(m):
                for j in range(n):
                    title = titles[i, j]
                    names.update({title: (i, j)})
        return names

    def set(self, index: Union[Axes, str, Iterable[int], slice, None] = None, **kwargs) -> None:
        r"""Set properties on selected axes.

        Parameters
        ----------
        index : matplotlib.axes.Axes, str, iterable, slice, or None, optional
            Target axes selector.
        **kwargs
            Properties passed to `Axes.set`.
        """
        if isinstance(index, Axes):
            index.set(**kwargs)
            return 1
        if isinstance(index, (str, Iterable)):
            index = [index]
        elif isinstance(index, slice):
            index = self.titles[index].flatten()
        elif index is None:
            index = self.titles.flatten()
        else:
            raise TypeError(f"[str, Iterable, slice, None] expected but {type(index)} received ...")
        for idx in index:
            ax = self[idx]
            ax.set(**kwargs)

    def set_title(self, y: float = 0.99, **kwargs) -> None:
        r"""Set titles for all managed axes.

        Parameters
        ----------
        y : float, default=0.99
            Vertical title position.
        **kwargs
            Additional keyword arguments passed to `Axes.set_title`.

        Examples
        --------
        >>> fp.set_title(y=0.9)
        """
        for title in self.links.keys():
            ax = self[title]
            ax.set_title(title, y=y, **kwargs)

    def ticklabel_format(
        self,
        index: Union[Axes, str, Iterable[int], slice, None] = None,
        style: Literal["sci", "scientific", "plain"] = "sci",
        scilimits: Iterable[int] = (0, 0),
        axis: str = "y",
        **kwargs,
    ):
        r"""Configure tick label formatting on selected axes.

        Parameters
        ----------
        index : matplotlib.axes.Axes, str, iterable, slice, or None, optional
            Target axes selector.
        axis : {'x', 'y', 'both'}, default: 'both'
            The axis to configure.  Only major ticks are affected.
        style : {'sci', 'scientific', 'plain'}
            Whether to use scientific notation.
            The formatter default is to use scientific notation.
            'sci' is equivalent to 'scientific'.
        scilimits : pair of ints (m, n)
            Scientific notation is used only for numbers outside the range
            10\ :sup:`m` to 10\ :sup:`n` (and only if the formatter is
            configured to use scientific notation at all).  Use (0, 0) to
            include all numbers.  Use (m, m) where m != 0 to fix the order of
            magnitude to 10\ :sup:`m`.
            The formatter default is ``axes.formatter.limits``.
        useOffset : bool or float
            If True, the offset is calculated as needed.
            If False, no offset is used.
            If a numeric value, it sets the offset.
            The formatter default is ``axes.formatter.useoffset``.
        useLocale : bool
            Whether to format the number using the current locale or using the
            C (English) locale.  This affects e.g. the decimal separator.  The
            formatter default is ``axes.formatter.use_locale``.
        useMathText : bool
            Render the offset and scientific notation in mathtext.
            The formatter default is ``axes.formatter.use_mathtext``.

        Raises
        ------
        AttributeError
            If the current formatter is not a `.ScalarFormatter`.
        """
        if isinstance(index, Axes):
            index.set(**kwargs)
            return 1
        if isinstance(index, (str, Iterable)):
            index = [index]
        elif isinstance(index, slice):
            index = self.titles[index].flatten()
        elif index is None:
            index = self.titles.flatten()
        else:
            raise TypeError(f"[str, Iterable, slice, None] expected but {type(index)} received ...")
        for idx in index:
            ax = self[idx]
            ax.ticklabel_format(style=style, scilimits=scilimits, axis=axis, **kwargs)

    def __iter__(self):
        return (ax.ax for ax in self.axes)

    def __getitem__(self, idx: Union[Iterable[int], str, Axes]):
        if not isinstance(idx, (Iterable, str, Axes)):
            raise KeyError(f"[Iterable, str, Axes] expected but {type(idx)} received ...")
        if isinstance(idx, Axes):
            return idx
        if isinstance(idx, str):
            idx = self.links[idx]
        ax = self.axes[idx]
        return ax.ax


class UnitPlot:
    r"""Grid-like plotting container.

    Parameters
    ----------
    shape : tuple[int, int], default=(1, 1)
        Axes grid shape as `(rows, cols)`.
    figsize : tuple[float, float], default=(1.5, 2.0)
        Per-axes figure size as `(height, width)`.
    titles : iterable, optional
        Titles used to access axes by name.
    sharey : bool, default=True
        Whether axes in the same row share y-axis.
    latex : bool, default=False
        Whether to keep LaTeX-related style settings enabled.
    dpi : int, default=500
        Figure DPI.
    projection : str, optional
        Matplotlib projection name.
    **kwargs
        Additional keyword arguments passed to `matplotlib.pyplot.figure`.

    Notes
    -----
    LaTeX must be installed in the local environment when `latex=True`.
    """

    def __init__(
        self,
        shape: Tuple[int, int] = (1, 1),
        figsize: Tuple[float, float] = (1.5, 2.0),
        titles: Optional[Iterable] = None,
        sharey: bool = True,
        latex: bool = False,
        dpi: int = 500,
        projection: Optional[str] = None,
        **kwargs,
    ):
        # the default settings
        plt.style.use(style_cfg.basic)
        if not latex:
            self.set_style("no-latex")
        for group, params in cfg["rc_params"].items():
            plt.rc(group, **params)

        figsize = (figsize[1] * shape[1], figsize[0] * shape[0])
        self.fig = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        self.axes = FreeAxes(self.fig, shape, titles, sharey, projection=projection)

    @property
    def colors(self):
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        return prop_cycle.by_key()["color"]

    @colors.setter
    def colors(self, palette: Literal["cool", "bright", "factor"] = "cool"):
        PALETTES = {
            "cool": COLORS,
            "bright": ["#16058b", "#6200AA", "#9E169D", "#CC4A74", "#EB7852", "#FCB431"],
            "factor": ["#021024", "#052659", "#4D77A6", "#5483B3", "#7DA0CA", "#C1E8FF"],
        }
        if isinstance(palette, str):
            palette = PALETTES[palette]
        plt.rcParams["axes.prop_cycle"] = cycler(marker=MARKERS, color=palette)

    @property
    def styles(self):
        r"""Return available Matplotlib and FreePlot style names."""
        return plt.style.available + list(style_cfg.keys())

    @property
    def rcParams(self):
        r"""Return current Matplotlib runtime settings."""
        return matplotlib.rcParams

    def set(self, index: Union[str, Iterable[int], slice, None] = None, **kwargs) -> None:
        r"""Set properties for selected axes.

        Parameters
        ----------
        index : str, iterable, slice, or None, optional
            Target axes selector.
        **kwargs
            Properties passed to `Axes.set`.
        """
        self.axes.set(index=index, **kwargs)

    def set_font(self, family: Literal["serif", "sans-serif"] = "sans-serif", size: int = 7):
        r"""Set the default Matplotlib font.

        Parameters
        ----------
        family : {"serif", "sans-serif"}, default="sans-serif"
            Font family.
        size : int, default=7
            Font size.
        """
        plt.rc("font", family=family, size=size)

    def set_style(self, style: Union[str, Iterable[str]]):
        r"""Apply one or more styles.

        Parameters
        ----------
        style : str or iterable of str
            Style name or names. Use `styles` to inspect available names.
        """
        styles = []
        if isinstance(style, str):
            styles += get_style(style)
        else:
            for item in style:
                styles += get_style(item)
        plt.style.use(styles)

    def set_scale(self, value: str = "symlog", index=(0, 0), axis="y", **kwargs) -> None:
        r"""Set axis scale.

        Parameters
        ----------
        value : {"log", "linear", "symlog", "logit"}, default="symlog"
            Scale name.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        axis : {"x", "y", "z"}, default="y"
            Axis name.
        **kwargs
            Additional properties passed to `set`.

        Examples
        --------
        >>> fp.set_scale(value="symlog", index=(0, 0))
        """
        kwargs["index"] = index
        kwargs[axis + "scale"] = value
        return self.set(**kwargs)

    def set_lim(self, lim: Iterable[float], index=(0, 0), axis="y", **kwargs):
        r"""Set axis limits.

        Parameters
        ----------
        lim : iterable of float
            Lower and upper axis limits.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        axis : {"x", "y", "z"}, default="y"
            Axis name.
        **kwargs
            Additional properties passed to `set`.

        Examples
        --------
        >>> fp.set_lim((0, 10), index=(0, 0), axis="y")
        >>> fp.set_lim((1, 5), index=(0, 0), axis="x")
        """
        kwargs["index"] = index
        kwargs[axis + "lim"] = lim
        return self.set(**kwargs)

    def set_label(self, label: str, index=(0, 0), axis="y", **kwargs):
        r"""Set an axis label.

        Parameters
        ----------
        label : str
            Axis label text.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        axis : {"x", "y", "z"}, default="y"
            Axis name.
        **kwargs
            Additional properties passed to `set`.

        Examples
        --------
        >>> fp.set_label("X", axis="x")
        >>> fp.set_label("Y", axis="y")
        """
        kwargs["index"] = index
        kwargs[axis + "label"] = label
        return self.set(**kwargs)

    def set_text(
        self, x: float, y: float, s: str, index=(0, 0), fontdict: Optional[Dict] = None, **kwargs
    ) -> matplotlib.text.Text:
        r"""Add text to an axes.

        Parameters
        ----------
        x : float
            X coordinate in data space.
        y : float
            Y coordinate in data space.
        s : str
            Text content.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        fontdict : dict, optional
            Text property overrides.
        **kwargs
            Additional keyword arguments passed to `Axes.text`.

        Returns
        -------
        matplotlib.text.Text
            Created text artist.

        Examples
        --------
        >>> fp.set_text(0.5, 0.5, s="GoGoGo", fontsize="large")
        """
        return self[index].text(x, y, s, fontdict, **kwargs)

    def set_arrow(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        width: float,
        head_width: float,
        index=(0, 0),
        color: str = "r",
        alpha: float = 0.5,
        **kwargs,
    ):
        r"""Add an arrow to an axes.

        Parameters
        ----------
        x : float
            Arrow start x coordinate.
        y : float
            Arrow start y coordinate.
        dx : float
            Arrow x offset.
        dy : float
            Arrow y offset.
        width : float
            Arrow shaft width.
        head_width : float
            Arrow head width.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        color : str, default="r"
            Arrow color.
        alpha : float, default=0.5
            Arrow opacity.
        **kwargs
            Additional keyword arguments passed to `Axes.arrow`.

        Returns
        -------
        matplotlib.patches.FancyArrow
            Created arrow artist.
        """
        return self[index].arrow(
            x, y, dx, dy, width=width, head_width=head_width, color=color, alpha=alpha, **kwargs
        )

    def set_title(self, y: float = 0.99) -> None:
        r"""Set configured titles on all axes.

        Parameters
        ----------
        y : float, default=0.99
            Vertical title position.

        Examples
        --------
        >>> fp.set_title(y=1.1)
        """
        self.axes.set_title(y=y)

    def set_ticks(
        self, values: Iterable, index=(0, 0), fmt: str = "%s", axis: str = "y", **kwargs
    ) -> Dict:
        r"""Set the values of ticks.

        Parameters
        ----------
        values : iterable
            Tick values.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        fmt : str, default="%s"
            Format string used to create tick labels.
        axis : {"x", "y", "z"}, default="y"
            Axis name.
        **kwargs
            Additional properties passed to `set`.

        Notes
        -----
        Passing an empty `values` sequence hides the selected axis labels.

        Examples
        --------
        >>> fp.set_ticks([0.1, 0.2, 0.3, 0.4, 0.5], fmt=".3f")
        >>> fp.set_ticks([])
        """
        labels = [fmt % value for value in values]
        kwargs["index"] = index
        kwargs[axis + "ticks"] = values
        kwargs[axis + "ticklabels"] = labels
        return self.set(**kwargs)

    def fill_between(
        self,
        x: Iterable,
        lower: Iterable,
        upper: Iterable,
        alpha: float = 0.5,
        linewidth: float = 0.0,
        index=(0, 0),
        **kwargs,
    ):
        r"""Fill the area between lower and upper curves.

        Parameters
        ----------
        x : iterable
            X coordinates.
        lower : iterable
            Lower y coordinates.
        upper : iterable
            Upper y coordinates.
        alpha : float, default=0.5
            Fill opacity.
        linewidth : float, default=0.0
            Boundary line width.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        **kwargs
            Additional keyword arguments passed to `Axes.fill_between`.

        Returns
        -------
        matplotlib.collections.PolyCollection
            Created fill artist.
        """
        return self[index].fill_between(x, lower, upper, alpha=alpha, linewidth=linewidth, **kwargs)

    def ticklabel_format(
        self,
        style: str = "sci",
        scilimits: Iterable[int] = (0, 0),
        index: Union[Axes, str, Iterable[int], slice, None] = (0, 0),
        axis: str = "y",
        **kwargs,
    ):
        r"""Configure the ScalarFormatter used by default for linear Axes.

        Parameters
        ----------
        style : {"sci", "scientific", "plain"}, default="sci"
            Tick label notation.
        scilimits : iterable of int, default=(0, 0)
            Scientific notation limits.
        index : matplotlib.axes.Axes, str, iterable, slice, or None, default=(0, 0)
            Target axes selector.
        axis : {"x", "y", "both"}, default="y"
            Axis name.
        **kwargs
            Additional keyword arguments passed to `Axes.ticklabel_format`.

        Examples
        --------
        >>> fp.ticklabel_format(style="sci", index=(0, 0))
        >>> fp.ticklabel_format(style="sci", index=None)
        """
        self.axes.ticklabel_format(
            index=index, style=style, scilimits=scilimits, axis=axis, **kwargs
        )

    def get_containers(self, index=(0, 0)) -> List[matplotlib.container.BarContainer]:
        r"""Return artist containers from an axes.

        Parameters
        ----------
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.

        Returns
        -------
        list[matplotlib.container.BarContainer]
            Containers such as bar plot groups.
        """
        ax = self[index]
        return ax.containers

    @inherit_from_matplotlib
    def get_facecolor(self, index=(0, 0)) -> Tuple[float]:
        """Get the facecolor of the Axes."""

    @inherit_from_matplotlib
    def get_legend(self, index=(0, 0)) -> Optional[matplotlib.legend.Legend]:
        """Return Legend instance or None if no legend"""

    @inherit_from_matplotlib
    def get_legend_handles_labels(self, index=(0, 0), legend_handler_map=None) -> Tuple[List]:
        """Return handles and labels for legend."""

    @inherit_from_matplotlib
    def get_lines(self, index=(0, 0)) -> Iterable[matplotlib.lines.Line2D]:
        """Return the lines contained in the Axes."""

    def get_patches(self, index=(0, 0)) -> List:
        """Return the patches in the Axes."""
        ax = self[index]
        return ax.patches

    @inherit_from_matplotlib
    def get_title(self, index=(0, 0)) -> str:
        """Return the title of the Axes."""

    @inherit_from_matplotlib
    def get_xaxis(self, index=(0, 0)) -> matplotlib.axis.Axis:
        """Return the XAxis."""

    @inherit_from_matplotlib
    def get_xlabel(self, index=(0, 0)) -> str:
        """Get the xlabel text string."""

    @inherit_from_matplotlib
    def get_xlim(self, index=(0, 0)) -> Tuple[float, float]:
        """Return the x-axis view limits."""

    @inherit_from_matplotlib
    def get_xscale(self, index=(0, 0)) -> str:
        """Return x-scale type."""

    @inherit_from_matplotlib
    def get_xticklabels(self, index=(0, 0)) -> Iterable[matplotlib.text.Text]:
        """Get the xaxis' tick labels."""

    @inherit_from_matplotlib
    def get_xticks(self, index=(0, 0)) -> np.ndarray:
        """Return the xaxis' tick locations in data coordinates."""

    @inherit_from_matplotlib
    def get_yaxis(self, index=(0, 0)) -> matplotlib.axis.Axis:
        """Return the YAxis."""

    @inherit_from_matplotlib
    def get_ylabel(self, index=(0, 0)) -> str:
        """Get the ylabel text string."""

    @inherit_from_matplotlib
    def get_ylim(self, index=(0, 0)) -> Tuple[float, float]:
        """Return the y-axis view limits."""

    @inherit_from_matplotlib
    def get_yscale(self, index=(0, 0)) -> str:
        """Return y-scale type."""

    @inherit_from_matplotlib
    def get_yticklabels(self, index=(0, 0)) -> Iterable[matplotlib.text.Text]:
        """Get the yaxis' tick labels."""

    @inherit_from_matplotlib
    def get_yticks(self, index=(0, 0)) -> np.ndarray:
        """Return the yaxis' tick locations in data coordinates."""

    @style_env
    def inset_axes(
        self,
        xlims: Iterable[float],
        ylims: Iterable[float],
        bounds: Iterable[float],
        *,
        style: Union[str, Iterable[str]] = None,
        index=(0, 0),
        patch_params: dict = {"edgecolor": "black", "linewidth": 0.7, "alpha": 0.5},
        line_params: dict = {"color": "gray", "linewidth": 0.5, "alpha": 0.7, "linestyle": "--"},
    ) -> Tuple[Axes, matplotlib.patches.Patch, Iterable]:
        r"""Add a zoomed inset axes.

        Parameters
        ----------
        xlims : iterable of float
            X limits for the inset view.
        ylims : iterable of float
            Y limits for the inset view.
        bounds : iterable of float
            Inset bounds as `(x0, y0, width, height)` in parent axes coordinates.
        style : str or iterable of str, optional
            Style names resolved by `style_env`.
        index : tuple[int, int] or str, default=(0, 0)
            Parent axes index or title.
        patch_params : dict, optional
            Properties applied to the inset rectangle patch.
        line_params : dict, optional
            Properties applied to connector lines.

        Returns
        -------
        tuple
            Inset axes, rectangle patch, and connector lines.

        Examples
        --------
        >>> fp = FreePlot((1, 1), (5, 4))
        >>> fp.lineplot([1, 2, 3], [4, 5, 6], label="a")
        >>> fp.lineplot([1, 2, 3], [3, 5, 7], label="b")
        >>> axins, patch, lines = fp.inset_axes(
        ...    xlims=(1.9, 2.1),
        ...    ylims=(4.9, 5.1),
        ...    bounds=(0.1, 0.7, 0.2, 0.2),
        ...    index=(0, 0),
        ...    style="line",
        ... )
        """
        axins = self[index].inset_axes(bounds)
        axins.set_xlim(xlims[0], xlims[1])
        axins.set_ylim(ylims[0], ylims[1])
        patch, lines = self[index].indicate_inset_zoom(axins, edgecolor="black")
        for name, value in patch_params.items():
            getattr(patch, "set_" + name)(value)
        for name, value in line_params.items():
            for line in lines:
                getattr(line, "set_" + name)(value)
        try:
            axins.get_legend().remove()
        except AttributeError:
            pass
        axins.set(xlabel=None, ylabel=None, title=None)
        return axins, patch, lines

    def set_figure_legend(
        self,
        x: float,
        y: float,
        ncol: int,
        index: Union[Tuple[int, int], str] = (0, 0),
        loc: str = "lower left",
        frameon: Optional[bool] = None,
        columnspacing: Optional[float] = None,
        title: Optional[str] = None,
        **kwargs,
    ) -> matplotlib.legend.Legend:
        r"""Set the legend relative to the figure.

        Parameters
        ----------
        x : float
            Legend anchor x coordinate in figure space.
        y : float
            Legend anchor y coordinate in figure space.
        ncol : int
            Number of legend columns.
        index : tuple[int, int] or str, default=(0, 0)
            Target axes index or title.
        loc : str, default="lower left"
            Legend location relative to the anchor.
        frameon : bool, optional
            Whether to draw a legend frame.
        columnspacing : float, optional
            Spacing between legend columns.
        title : str, optional
            Legend title.
        **kwargs
            Additional keyword arguments passed to `Axes.legend`.

        Notes
        -----
        Figure-level legends can conflict with tight layout.

        Returns
        -------
        matplotlib.legend.Legend
            Created legend.

        Examples
        --------
        >>> fp.set_figure_legend(0.3, 0.9, ncol=3)
        >>> fp.savefig(tight_layout=False)
        """
        return self[index].legend(
            bbox_to_anchor=(x, y),
            loc=loc,
            bbox_transform=plt.gcf().transFigure,
            ncol=ncol,
            frameon=frameon,
            columnspacing=columnspacing,
            title=title,
            **kwargs,
        )

    def subplots_adjust(
        self,
        left: Optional[float] = None,
        bottom: Optional[float] = None,
        right: Optional[float] = None,
        top: Optional[float] = None,
        wspace: Optional[float] = None,
        hspace: Optional[float] = None,
    ) -> None:
        r"""Adjust subplot layout parameters.

        Parameters
        ----------
        left : float, optional
            Left side of the subplots.
        bottom : float, optional
            Bottom side of the subplots.
        right : float, optional
            Right side of the subplots.
        top : float, optional
            Top side of the subplots.
        wspace : float, optional
            Width reserved for space between subplots.
        hspace : float, optional
            Height reserved for space between subplots.
        """
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

    @staticmethod
    def imread(filename: str, fmt: Optional[str] = None):
        r"""Load an image.

        Parameters
        ----------
        filename : str
            Image path.
        fmt : str, optional
            Image format.

        Returns
        -------
        np.ndarray
            Loaded image data.
        """
        return plt.imread(filename, fmt)

    @staticmethod
    def convert(img: np.ndarray, cur_fmt: str, nxt_fmt: str, **kwargs):
        r"""Convert an image between color spaces.

        Parameters
        ----------
        img : np.ndarray
            Image data.
        cur_fmt : str
            Current color space.
        nxt_fmt : str
            Target color space.
        **kwargs
            Additional keyword arguments passed to `skimage.color` conversion.

        Returns
        -------
        np.ndarray
            Converted image.
        """
        from skimage import color

        available = (
            "gray",
            "hed",
            "hsv",
            "lab",
            "label",
            "rgb",
            "rgba",
            "rgbcie",
            "xyz",
            "ycbcr",
            "ycbdr",
            "yiq",
            "ypbpr",
            "yuv",
        )
        cur_fmt, nxt_fmt = cur_fmt.lower(), nxt_fmt.lower()
        assert cur_fmt in available, f"current format is not in {available}"
        assert nxt_fmt in available, f"next format is not in {available}"
        trans = "2".join((cur_fmt, nxt_fmt))
        return getattr(color, trans)(img, **kwargs)

    def savefig(
        self,
        filename: str,
        close_fig: bool = True,
        tight_layout: bool = False,
        bbox_inches: str = "tight",
        **kwargs,
    ) -> None:
        r"""Save the figure.

        Parameters
        ----------
        filename : str
            Output path.
        close_fig : bool, default=True
            Whether to close the figure after saving.
        tight_layout : bool, default=False
            Whether to call `matplotlib.pyplot.tight_layout` before saving.
        bbox_inches : str, default="tight"
            Bounding box mode passed to `Figure.savefig`.
        **kwargs
            Additional keyword arguments passed to `Figure.savefig`.

        Notes
        -----
        `tight_layout` will conflict with other settings sometimes.
        """
        if tight_layout:
            plt.tight_layout()
        self.fig.savefig(filename, bbox_inches=bbox_inches, **kwargs)
        if close_fig:
            self.close()

    def close(self) -> None:
        r"""Close the figure."""
        plt.close(self.fig)

    def show(self, *args, tight_layout: bool = False, **kwargs):
        r"""Show the figure.

        Parameters
        ----------
        *args
            Positional arguments passed to `matplotlib.pyplot.show`.
        tight_layout : bool, default=False
            Whether to call `matplotlib.pyplot.tight_layout` before showing.
        **kwargs
            Keyword arguments passed to `matplotlib.pyplot.show`.
        """
        if tight_layout:
            plt.tight_layout()
        return plt.show(*args, **kwargs)

    def __getitem__(self, index: Union[Iterable[int], str, Axes]) -> Union[Axes, Axes3D]:
        r"""Get Axes.

        Parameters
        ----------
        index : iterable of int, str, or matplotlib.axes.Axes
            Axes selector.

        Returns
        -------
        matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D
            Selected axes.
        """
        return self.axes[index]
