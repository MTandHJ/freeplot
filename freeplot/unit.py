

from typing import List, Tuple, Optional, Dict, Union, Iterable, Literal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from cycler import cycler

from .config import cfg, style_cfg, COLORS, MARKERS
from .utils import inherit_from_matplotlib, get_style, style_env


class UnitAX:
    """Single Axes."""

    def __init__(
        self, axes: 'FreeAxes', 
        position: matplotlib.gridspec.GridSpec, 
        anchor: Optional['UnitAX'] = None,
        sharey: bool = True, **kwargs
    ):
        """
        Parameters:
        -----------

        axes: FreeAxes.
        position: The grid position in the figure.
        anchor: 
            - `UnitAX`: This Axes will follow the anchor.
            - `None`: This Axes will not follow any anchor.

        sharey: `True`: The anchor will share y axis with current Axe.
        **kwargs: other kwargs for `fig.add_subplot`
        """
        self.axes = axes
        self.position = position
        self.anchor = anchor
        self.sharey = sharey
        self.kwargs = kwargs
        self.__ax = None

    @property
    def ax(self):
        """Return Axes.

        Notes:
        ------

        The Axes will be then created after this method is called.
        This operation is necessary for users to specify 'style' during plotting.

        Returns:
        ---
        Axes
        """
        if self.__ax is None:
            if not self.sharey or self.anchor is None:
                self.__ax = self.axes.fig.add_subplot(
                    self.position, **self.kwargs
                )
            else:
                self.__ax = self.axes.fig.add_subplot(
                    self.position, sharey=self.anchor.ax, **self.kwargs
                )
                plt.setp(self.__ax.get_yticklabels(), visible=False)
        return self.__ax


class FreeAxes:
    """Collection of Axes."""

    def __init__(
        self, fig: matplotlib.figure.Figure, shape: Tuple[int, int], 
        titles: Optional[Iterable] = None,
        sharey: bool = True, projection: Optional[str] = None,
    ):
        r"""
        FreeAxes will create grids within the figure for every Axe.
        The first Axe in each row could be the the anchor for sharing y axis.

        Parameters:
        -----------
        fig: Figure.
        shape: (int, int)
            (#rows, #cols)
        titles: Optional[Iterable[str]]
            The titles for each Axes.
        sharey: bool
            Whether sharing the y axis.
        projection: str
            Sometimes it will be useful, like in case of '3d'.
        """
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
        """Link titles to corresponding Axes."""
        m, n = self.axes.shape
        names = dict()
        if titles is None:
            for i in range(m):
                for j in range(n):
                    s = "(" + chr(i * n + j + 97) + ")"
                    names.update({s:(i, j)})
        else:
            for i in range(m):
                for j in range(n):
                    title = titles[i, j]
                    names.update({title:(i, j)})
        return names

    def set(self, index: Union[Axes, str, Iterable[int], slice, None] = None, **kwargs) -> None:
        """Set properties."""
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

    def set_title(self, y: float = .99, **kwargs) -> None:
        """
        Set titles for all axes.

        Examples:
        ---------
        >>> fp.set_title(y=0.9)
        """
        for title in self.links.keys():
            ax = self[title]
            ax.set_title(title, y=y, **kwargs)

    def ticklabel_format(
        self, 
        index: Union[Axes, str, Iterable[int], slice, None] = None,
        style: Literal['sci', 'scientific', 'plain'] = 'sci', 
        scilimits: Iterable[int] = (0, 0),
        axis: str = 'y', **kwargs
    ):
        r"""
        Configure the `.ScalarFormatter` used by default for linear Axes.

        If a parameter is not set, the corresponding property of the formatter
        is left unchanged.

        Parameters:
        -----------
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
            The formatter default is :rc:`axes.formatter.limits`.
        useOffset : bool or float
            If True, the offset is calculated as needed.
            If False, no offset is used.
            If a numeric value, it sets the offset.
            The formatter default is :rc:`axes.formatter.useoffset`.
        useLocale : bool
            Whether to format the number using the current locale or using the
            C (English) locale.  This affects e.g. the decimal separator.  The
            formatter default is :rc:`axes.formatter.use_locale`.
        useMathText : bool
            Render the offset and scientific notation in mathtext.
            The formatter default is :rc:`axes.formatter.use_mathtext`.

        Raises:
        -------
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
    r"""Plotting grid-like Axes.

    Parameters:
    -----------
    shape: (#rows, #cols)
    figsize: (height, width) for each ax
        So the real figsize is (height * row, width * col).
    titles: Optional[Iterable[str]]
        Titles for Axes.
    sharey: bool 
        Whether sharing the y axis.
    latex: bool
        `False`: set_style('no-latex').
    projection: Sometimes it will be useful, like in case of '3d'.
    **kwargs: other kwargs for `plt.figure`.
    
    Notes:
    ------
    Please make sure your computer has installed Latex already before calling 'latex=True' !
    """

    def __init__(
        self, 
        shape: Tuple[int, int] = (1, 1), 
        figsize: Tuple[float, float] = (1.5, 2.), 
        titles: Optional[Iterable] = None,
        sharey: bool = True,
        latex: bool = False,
        dpi: int = 500,
        projection: Optional[str] = None,
        **kwargs
    ):
        # the default settings
        plt.style.use(style_cfg.basic)
        if not latex:
            self.set_style('no-latex')
        for group, params in cfg['rc_params'].items():
            plt.rc(group, **params)

        figsize = (figsize[1] * shape[1], figsize[0] * shape[0])
        self.fig = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        self.axes = FreeAxes(self.fig, shape, titles, sharey, projection=projection)

    @property
    def colors(self):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        return prop_cycle.by_key()['color']

    @colors.setter
    def colors(self, palette: Literal['cool', 'bright', 'factor'] = 'cool'):
        PALETTES = {
            'cool': COLORS,
            'bright': [
                "#16058b",
                "#6200AA",
                "#9E169D",
                "#CC4A74",
                "#EB7852",
                "#FCB431"
            ],
            'factor': [
                "#021024",
                "#052659",
                "#4D77A6",
                "#5483B3",
                "#7DA0CA",
                "#C1E8FF"
            ]
        }
        if isinstance(palette, str):
            palette = PALETTES[palette]
        plt.rcParams['axes.prop_cycle'] = cycler(marker=MARKERS, color=palette)
    
    @property
    def styles(self):
        """Return available styles."""
        return plt.style.available + list(style_cfg.keys())

    @property
    def rcParams(self):
        """Return current settings."""
        return matplotlib.rcParams

    def set(self, index: Union[str, Iterable[int], slice, None] = None, **kwargs) -> None:
        """Set properties for the Axes of 'index'."""
        self.axes.set(index=index, **kwargs)

    def set_font(
        self,
        family: Literal['serif', 'sans-serif'] = 'sans-serif',
        size: int = 7
    ):
        r"""
        Parameters:
        -----------
        family: 'serif' or 'sans-serif'
        size: int
            font size
        """
        plt.rc(
            'font', family=family, size=size
        )

    def set_style(self, style: Union[str, Iterable[str]]):
        """Set a style. You can calling self.styles to check what is available."""
        styles = []
        if isinstance(style, str):
            styles += get_style(style)
        else:
            for item in style:
                styles += get_style(item)
        plt.style.use(styles)

    def set_scale(self, value: str = 'symlog', index=(0, 0), axis='y', **kwargs) -> None:
        """Convert the axis to other formats for readability.

        Parameters:
        ---
        value: 'log'|'linear'|'symlog'|'logit'
        axis: 'x'|'y'|'z'

        Examples:
        ---
        >>> fp.set_scale(value='symlog', index=(0, 0))
        """
        kwargs['index'] = index
        kwargs[axis + 'scale'] = value
        return self.set(**kwargs)

    def set_lim(self, lim: Iterable[float], index=(0, 0), axis='y', **kwargs):
        r"""
        Set the range for the axis.

        Parameters:
        -----------
        lim: (low, high)
        axis: 'x'|'y'|'z'

        Examples:
        ---------
        >>> fp.set_lim((0, 10), index=(0, 0), axis='y')
        >>> fp.set_lim((1, 5), index=(0, 0), axis='x')
        """
        kwargs['index'] = index
        kwargs[axis + 'lim'] = lim
        return self.set(**kwargs)

    def set_label(self, label: str, index=(0, 0), axis='y', **kwargs):
        r"""
        Set a label for the axis.

        Parameters:
        -----------
        label: str
        axis: 'x'|'y'|'z'

        Examples:
        ---------
        >>> fp.set_label('X2X', axis='x')
        >>> fp.set_label('Y2Y', axis='y')
        """
        kwargs['index'] = index
        kwargs[axis + 'label'] = label
        return self.set(**kwargs)

    def set_text(
        self, x: float, y: float, s: str, 
        index=(0, 0), fontdict: Optional[Dict] = None,
        **kwargs
    ) -> matplotlib.text.Text:
        r"""
        Add a text to the ax.

        Parameters:
        -----------
        (x, y): Data coordinates.
        s: The text.
        fontdict: A dictionary to override the default text properties.
        **kwargs: other kwargs for `text`
            - fontsize: positive interger or 'xx-small', 'x-small', 'small', 'big'
            - alpha: ...
            - ...

        Examples:
        ---------
        >>> fp.set_text(0.5, 0.5, s='GoGoGo', fontsize='big)
        """
        return self[index].text(x, y, s, fontdict, **kwargs)

    def set_arrow(
        self, 
        x: float, y: float,
        dx: float, dy: float,
        width: float, head_width: float,
        index = (0, 0), color: str = 'r', alpha: float = 0.5,
        **kwargs
    ):
        return self[index].arrow(
            x, y, dx, dy,
            width=width, head_width=head_width,
            color=color, alpha=alpha, **kwargs
        )

    def set_title(self, y: float = .99) -> None:
        r"""
        Set titles for Axes.
        
        Parameters:
        -----------
        y: The height from the bottom.

        Examples:
        ---------
        >>> fp.set_title(y=1.1)
        """
        self.axes.set_title(y=y)

    def set_ticks(self, values: Iterable, index=(0, 0), fmt: str = "%s", axis: str = 'y', **kwargs) -> Dict:
        r"""Set the values of ticks.
        
        Parameters:
        ---

        values: The values of the ticks.
        fmt: Display format of values, for example, 
            0.1234 with the format of "%2f" will be converted to 0.12
        axis: 'x'|'y'|'z'

        Examples:
        ---------
        >>> fp.set_ticks([0.1, 0.2, 0.3, 0.4, 0.5], fmt=".3f")
        >>> fp.set_ticks([])
        
        Notes:
        ------
        set_ticks([], axis='y') == self.get_yaxis().set_visible(False)
        """
        labels = [fmt%value for value in values]
        kwargs['index'] = index
        kwargs[axis + 'ticks'] = values
        kwargs[axis + 'ticklabels'] = labels
        return self.set(**kwargs)

    def fill_between(
        self,
        x: Iterable,
        lower: Iterable, upper: Iterable,
        alpha: float = 0.5, linewidth: float = 0.,
        index = (0, 0),
        **kwargs
    ):
        r"""
        Fill between lower and upper.

        Parameters:
        -----------
        x: array
            The x coordinates of the nodes defining the curves.
        lower: array or scalar
            The y coordinates of the nodes defining the first curve.
        upper: array or scalar
            The y coordinates of the nodes defining the second curve.
        
        Returns:
        --------
        PolyCollection
        """
        return self[index].fill_between(
            x, lower, upper, alpha=alpha, linewidth=linewidth,
            **kwargs
        )
    
    def ticklabel_format(
        self, style: str = 'sci', scilimits: Iterable[int] = (0, 0),
        index: Union[Axes, str, Iterable[int], slice, None] = (0, 0), 
        axis: str = 'y', **kwargs
    ):
        r"""Configure the ScalarFormatter used by default for linear Axes.
        See [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html?highlight=ticklabel_format#matplotlib.axes.Axes.ticklabel_format) for details.

        Parameters:
        -----------
        style: 'sci'|'scientific'|'plain'
            - `plain`: 7000000.
            - `sci`: :math: `7 \times 10^6`
        scilimits: (m, n)
            Only numbers between 10^m and 10^n will be transformed to scientific notation.
            - `(0, 0)`: All numbers will be transformed.
        index:
            - `None`: Apply it for all Axes.
            - `slice`: ndarray slice supported.
        axis: 'x'|'y'|'both'
        **kwargs: other kwargs for `ticklabel_format`

        Examples:
        ---------
        >>> fp.ticklabel_format(style='sci', index=(0, 0))
        >>> fp.ticklabel_format(style='sci', index=None)
        """
        self.axes.ticklabel_format(index=index, style=style, scilimits=scilimits, axis=axis, **kwargs)

    def get_container(self, index=(0, 0)) -> List[matplotlib.container.BarContainer]:
        """Return containers that collect semantically related Artists such as the bars of a bar plot."""
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
        xlims: Iterable[float], ylims: Iterable[float], bounds: Iterable[float],
        *, style: Union[str, Iterable[str]] = None, index=(0, 0),
        patch_params: dict = {'edgecolor':'black', 'linewidth':.7, 'alpha':.5},
        line_params: dict = {'color':'gray', 'linewidth':.5, 'alpha':.7, 'linestyle':'--'}
    ) -> Tuple[Axes, matplotlib.patches.Patch, Iterable]:
        r"""
        Add a child inset Axes to this existing Axes.
        See [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.inset_axes.html?highlight=inset_axes#matplotlib.axes.Axes.inset_axes) for details.

        Parameters:
        -----------
        xlims|ylims: (l, r)|(b, t) 
            Determines the retangle region from l to r and from b to t.
        Bounds: (x0, y0, width, height) 
            The new inseted ax located in rectangle (x0, y0) to (x0 + width, y0 + height).
            Note that these values are according to the original ax, so they should be in [0, 1].
        style: You must choose the same style as keep consistent with the original ax.
        patch_params: You could specific the patch by passing a style dict;
        line_params: You could specific the line by passing a style dict.

        Examples:
        ---------
        >>> fp = FreePlot((1, 1), (5, 4))
        >>> fp.lineplot([1, 2, 3], [4, 5, 6], label='a')
        >>> fp.lineplot([1, 2, 3], [3, 5, 7], label='b')
        >>> axins, patch, lines = fp.inset_axes(
        ...    xlims=(1.9, 2.1),
        ...    ylims=(4.9, 5.1),
        ...    bounds=(0.1, 0.7, 0.2, 0.2),
        ...    index=(0, 0),
        ...    style='line' # The style should be consistent with the style of Axes[0, 0].
        ... )
        >>> fp.lineplot([1, 2, 3], [4, 5, 6], index=axins)
        >>> fp.lineplot([1, 2, 3], [3, 5, 7], index=axins)

        Returns: 
        --------
        Axes, Patch, Lines
        """
        axins = self[index].inset_axes(bounds)
        axins.set_xlim(xlims[0], xlims[1])
        axins.set_ylim(ylims[0], ylims[1])
        patch, lines = self[index].indicate_inset_zoom(axins, edgecolor='black')
        for name, value in patch_params.items():
            getattr(patch, 'set_' + name)(value)
        for name, value in line_params.items():
            for line in lines:
                getattr(line, 'set_' + name)(value)
        try:
            axins.get_legend().remove()
        except AttributeError:
            pass
        axins.set(xlabel=None, ylabel=None, title=None)
        return axins, patch, lines

    def legend(
        self, 
        x: float, y: float, ncol: int, 
        index: Union[int, str] = (0, 0), 
        loc: str = "lower left",
        frameon: Optional[bool] = None,
        columnspacing: Optional[float] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> None:
        r"""Set the legend relative to the figure.
        See [here](https://matplotlib.org/stable/api/legend_api.html?highlight=legend#module-matplotlib.legend) for details.
        
        Parameters:
        -----------
        (x, y): Coordinates relative to the figure.
        ncol: Split legends into 'ncol' columns.
        loc: 'upper left', 'upper right', 'lower left', 'lower right'
            The location of the legend.
        **kwargs: other kwargs for `legend`
            - handles: A list of Artists (lines, patches) to be added to the legend
            - labels: A list of labels to show next to the artists. 
                    The length of handles and labels should be the same. 
                    If they are not, they are truncated to the smaller of both lengths.
            - fontsize: int or 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
            - labelcolor: str or list
            - markerscale: float 
                    The relative size of legend markers compared with the originally drawn ones
            - facecolor: 'inherit' or color
            - edgecolor: 'inherit' or color
            - ...

        Notes:
        ------
        When you calling fp.legend(), please close the tight_layout in show() or savefig() !

        Examples:
        ---------
        >>> fp.legend(0.3, 0.9, ncol=3) # Add legend to the top of the figure.
        >>> fp.savefig(tight_layout=False) # !!!
        >>> # The following operation will not conflict with tight_layout.
        >>> fp[0, 0].legend() # Add legend in the Axes[0, 0].
        """
        return self[index].legend(
            bbox_to_anchor=(x, y), loc=loc,
            bbox_transform=plt.gcf().transFigure, ncol=ncol,
            frameon=frameon, columnspacing=columnspacing, title=title,
            **kwargs
        )

    def subplots_adjust(
        self,
        left: Optional[float] = None, 
        bottom: Optional[float] = None, 
        right: Optional[float] = None, 
        top: Optional[float] = None, 
        wspace: Optional[float] = None, 
        hspace: Optional[float] = None
    ) -> None:
        r"""
        Adjust the subplot layout parameters.
        See https://matplotlib.org/stable/api/figure_api.html?highlight=subplots_adjust#matplotlib.figure.Figure.subplots_adjust for details.
        """
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

    @staticmethod
    def imread(filename: str, fmt: Optional[str] = None):
        """Load Img."""
        return plt.imread(filename, fmt)

    @staticmethod
    def convert(img: np.ndarray, cur_fmt: str, nxt_fmt: str, **kwargs):
        r"""
        Convert the image into another type.
            
        Parameters:
        -----------
        img: Image
        cur|nxt_fmt: gray, hed, hsv, lab, label, rgb, rgba, rgbcie, 
            xyz, ycbcr, ycbdr, yiq, ypbpr, yuv
        **kwargs: other kwargs for `skimage.color`
        """
        from skimage import color
        available = ('gray', 'hed', 'hsv', 'lab', 'label', 'rgb', 'rgba', 'rgbcie',
                    'xyz', 'ycbcr', 'ycbdr', 'yiq', 'ypbpr', 'yuv')
        cur_fmt, nxt_fmt = cur_fmt.lower(), nxt_fmt.lower()
        assert cur_fmt in available, f"current format is not in {available}"
        assert nxt_fmt in available, f"next format is not in {available}"
        trans = '2'.join((cur_fmt, nxt_fmt))
        return getattr(color, trans)(img, **kwargs)

    def savefig(
        self, filename: str, 
        close_fig: bool = True,
        tight_layout: bool = False,
        bbox_inches: str = 'tight', 
        **kwargs
    ) -> None:
        r"""
        Save the figure and close it.

        Parameters:
        -----------
        close_fig: Close the figure to release memory if True (suggested).
        tight_layout: `True`: wspace, hspace will be no use.
        **kwargs: other kwargs for `plt.savefig`
        
        Notes:
        ------
        `tight_layout` will conflict with other settings sometimes.
        """
        if tight_layout:
            plt.tight_layout()
        self.fig.savefig(
            filename,
            bbox_inches=bbox_inches,
            **kwargs
        )
        if close_fig:
            self.close()

    def close(self) -> None:
        """Close the figure."""
        plt.close(self.fig)
    
    def show(self, *args, tight_layout: bool = False, **kwargs):
        """Show the figure."""
        if tight_layout:
            plt.tight_layout()
        return plt.show(*args, **kwargs)

    def __getitem__(self, index: Union[Iterable[int], str, Axes]) -> Union[Axes, Axes3D]:
        r"""Get Axes.

        Parameters:
        -----------
        index: titles or Tuple of index
            - `str`: return the Axes whose title is `str`
            - `(int, int)`: return the Axes located in `(int, int)`
            - `Axes`: return `Axes`

        Returns:
        --------
        Axes, Axes3D
        """
        return self.axes[index]