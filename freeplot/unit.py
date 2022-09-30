



from typing import List, Tuple, Optional, Dict, Union, Iterable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import os

from .config import cfg, style_cfg, COLORS
from .utils import inherit_from_matplotlib, get_style, style_env



class UnitAX:
    """Single Axes.
    """

    def __init__(
        self, axes: 'FreeAxes', 
        position: matplotlib.gridspec.GridSpec, 
        anchor: Optional['UnitAX'] = None,
        sharey: bool = True, **kwargs
    ):
        """
        Args:
            axes: FreeAxes
            position: the grid position in the figure
        Kwargs:
            anchor: this Axes will follow the anchor if not None
            sharey: the anchor will share y axis with current axe if True

        **kwargs: other kwargs for fig.add_subplot()
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

        Note that only this method be called and the Axes will be then created.  
        This operation is necessary for users to specify 'style' during plotting.
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
    """Collection of Axes.
    """

    def __init__(
        self, fig: matplotlib.figure.Figure, shape: Tuple[int, int], 
        titles: Optional[Iterable] = None,
        sharey: bool = True, projection: Optional[str] = None,
    ):
        """
        Args:
            fig: figure
            shape: (row, col)
        Kwargs:
            titles: for each axe
            sharey: share y axis  if true
            projection: sometimes will be useful, like in case of '3d'

        FreeAxes will create grids in the figure for every axe.
        The first axe in each row is the the anchor that may share y axis for its followers.
        """
        assert len(shape) == 2, "Only grid-like distribution Axes are supported, so 'shape' should be Tuple[int, int]"

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

    def set_title(self, y: float = .99) -> None:
        for title in self.links.keys():
            ax = self[title]
            ax.set_title(title, y=y)

    def ticklabel_format(
        self, 
        index: Union[Axes, str, Iterable[int], slice, None] = None,
        style: str = 'sci', 
        scilimits: Iterable[int] = (0, 0),
        axis: str = 'y', **kwargs
    ):
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
    """Plot with Axes of grid-like distribution"""

    def __init__(
        self, 
        shape: Tuple[int, int] = (1, 1), 
        figsize: Tuple[float, float] = (2.2, 2), 
        titles: Optional[Iterable] = None,
        sharey: bool = True,
        latex: bool = False,
        projection: Optional[str] = None,
        **kwargs: "other kwargs of plt.subplots"
    ):
        """
        Kwargs:
            shape: (row, col)
            figsize: (width (col), height (row))
            titles: titles for each Axe
            sharey: Axes in each row will share the same y axis if True
            latex: set_style('no-latex') if False
            projection: will be useful if '3d' is required.
        **kwargs: other kwargs for plt.figure
        
        NOTE: Please make sure your computer has installed Latex already before calling 'latex=True' !
        """
        # the default settings
        plt.style.use(style_cfg.basic)
        for group, params in cfg['rc_params'].items():
            plt.rc(group, **params)
        if not latex:
            self.set_style('no-latex')

        self.fig = plt.figure(figsize=figsize, **kwargs)
        self.axes = FreeAxes(self.fig, shape, titles, sharey, projection=projection)
    
    @property
    def styles(self):
        """Return available styles."""
        return plt.style.available + list(style_cfg.keys())

    @property
    def colors(self):
        """Return available colors."""
        return COLORS
    
    @property
    def rcParams(self):
        """Return current settings."""
        return matplotlib.rcParams

    def set(self, index: Union[str, Iterable[int], slice, None] = None, **kwargs) -> None:
        """Set properties for the Axes of 'index'."""
        self.axes.set(index=index, **kwargs)

    def set_style(self, style: Union[str, Iterable[str]]):
        """Set a style. You can calling self.styles to check what is available."""
        styles = []
        if isinstance(style, str):
            styles += get_style(style)
        else:
            for item in style:
                styles += get_style(item)
        plt.style.use(styles)

    def set_scale(self, value: str = 'symlog', index=(0, 0), axis='y') -> None:
        """Convert the axis to other formats for readability.

        Kwargs:
            value: 'log'|'linear'|'symlog'|'logit'
            axis: 'x'|'y'|'z'
        """
        kwargs = dict()
        kwargs['index'] = index
        kwargs[axis + 'scale'] = value
        return self.set(**kwargs)

    def set_lim(self, lim: Iterable[float], index=(0, 0), axis='y'):
        """Set the range for the axis.

        Kwargs:
            lim: (low, high)
            axis: 'x'|'y'|'z'
        """
        kwargs = dict()
        kwargs['index'] = index
        kwargs[axis + 'lim'] = lim
        return self.set(**kwargs)

    def set_label(self, label: str, index=(0, 0), axis='y'):
        """ Set a label for the axis.

        Kwargs:
            label: str
            axis: 'x'|'y'|'z'
        """
        kwargs = dict()
        kwargs['index'] = index
        kwargs[axis + 'label'] = label
        return self.set(**kwargs)

    def set_text(
        self, x: float, y: float, s: str, 
        index=(0, 0), fontdict: Optional[Dict] = None,
        **kwargs
    ) -> matplotlib.text.Text:
        """Add a text in the Axes.

        Args:
            (x, y): data coordinates;
            s: the text
        Kwargs:
            fontdict: a dictionary to override the default text properties.
        **kwargs: other kwargs for text()
            fontsize: positive interger or 'xx-small', 'x-small', 'small', 'big' ...
            alpha: ...
            ...
        """
        return self[index].text(x, y, s, fontdict, **kwargs)

    def set_title(self, y: float = .99) -> None:
        """Set titles for Axes.
        
        Kwargs:
            y: the height from the bottom.
        """
        self.axes.set_title(y=y)

    def set_ticks(self, values: Iterable, index=(0, 0), fmt: str = "%2f", axis: str = 'y') -> Dict:
        """Set ticks.
        
        Args:
            values: the values of the ticks
        Kwargs:
            fmt: display format of values, for example, 
                0.1234 with the format of "%2f" will be converted to 0.12
            axis: 'x'|'y'|'z'
        
        NOTE: set_ticks([], axis='y') == self.get_yaxis().set_visible(False)
        """
        labels = [fmt%value for value in values]
        kwargs = dict()
        kwargs['index'] = index
        kwargs[axis + 'ticks'] = values
        kwargs[axis + 'ticklabels'] = labels
        return self.set(**kwargs)

    
    def ticklabel_format(
        self, style: str = 'sci', scilimits: Iterable[int] = (0, 0),
        index=None, axis: str = 'y', **kwargs
    ):
        """Configure the ScalarFormatter used by default for linear Axes.
        See https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html?highlight=ticklabel_format#matplotlib.axes.Axes.ticklabel_format for details.

        Kwargs:
            style: 'sci'|'scientific'|'plain', 'plain' is 7000000 actually.
            scilimits: (m, n). Only numbers between 10^m and 10^n will be
                    transformed to scientific notation. The default settings (0, 0)
                    means all numbers will be transformed.
            axis: 'x'|'y'|'both'

        **kwargs: other kwargs for ticklabel_format

        When encountering very big numbers like 7000000,
        it will be ugly to display them on the ticks directly.
        By calling ticklabel_format with style=='sci', it will become 7 x 10^6.
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
        """Add a child inset Axes to this existing Axes.
        See https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.inset_axes.html?highlight=inset_axes#matplotlib.axes.Axes.inset_axes for details.

        Args:
            xlims|ylims: (l, r)|(b, t), to determine the retangle region from l to r and from b to t;
            bounds: (x0, y0, width, height), the new inseted ax located in rectangle (x0, y0) to (x0 + width, y0 + height).
                    Note that these values are according to the original ax, so they should be in [0, 1].
        Kwargs:
            style: You must choose the same style as keep consistent with the original ax.
            patch_params: You could specific the patch by passing a style dict;
            line_params: You could specific the line by passing a style dict.

        Returns: Axes, Patch, Lines
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
        **kwargs
    ) -> None:
        """Set the legend relative to the figure.
        See https://matplotlib.org/stable/api/legend_api.html?highlight=legend#module-matplotlib.legend for details.
        
        Args:
            (x, y): coordinates relative to the figure
            ncol: split legends into 'ncol' columns
        Kwargs:
            loc: the location of the legend, 'upper left', 'upper right', 'lower left', 'lower right'
        **kwargs: other kwargs for legend()
            handles: a list of Artists (lines, patches) to be added to the legend
            labels: a list of labels to show next to the artists. 
                    The length of handles and labels should be the same. 
                    If they are not, they are truncated to the smaller of both lengths.
            fontsize: int or 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
            labelcolor: str or list
            markerscale: float, the relative size of legend markers compared with the originally drawn ones
            facecolor: 'inherit' or color
            edgecolor: 'inherit' or color
            ...
        
        NOTE: when you calling fp.legend(), please close the tight_layout in show() or savefig() !
        """
        self[index].legend(bbox_to_anchor=(x, y), loc=loc,
        bbox_transform=plt.gcf().transFigure, ncol=ncol, **kwargs)

    def subplots_adjust(
        self,
        left: Optional[float] = None, 
        bottom: Optional[float] = None, 
        right: Optional[float] = None, 
        top: Optional[float] = None, 
        wspace: Optional[float] = None, 
        hspace: Optional[float] = None
    ) -> None:
        """Adjust the subplot layout parameters.
        See https://matplotlib.org/stable/api/figure_api.html?highlight=subplots_adjust#matplotlib.figure.Figure.subplots_adjust for details.
        """
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

    @staticmethod
    def imread(filename: str, fmt: Optional[str] = None):
        """Load Img."""
        return plt.imread(filename, fmt)

    @staticmethod
    def convert(img: np.ndarray, cur_fmt: str, nxt_fmt: str, **kwargs):
        """ Convert the image into another type.

        Args:
            img: ndarry
            cur|nxt_fmt: gray, hed, hsv, lab, label, rgb, rgba, rgbcie, 
                xyz, ycbcr, ycbdr, yiq, ypbpr, yuv
        **kwargs: other kwargs for skimage.color
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
        tight_layout: bool = True,
        bbox_inches: str = 'tight', 
        **kwargs: "other kwargs of plg.savefig"
    ) -> None:
        """ Save the figure and close it.

        Kwargs:
            close_fig: close the figure to release memory if True (suggested)
            tight_layout: wspace, hspace will be no use if True
        
        NOTE: tight_layout will conflict with other settings sometimes.
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
    
    def show(self, *args, tight_layout: bool = True, **kwargs):
        """Show the figure."""
        if tight_layout:
            plt.tight_layout()
        return plt.show(*args, **kwargs)

    def __getitem__(self, idx: Union[str, Tuple[int]]) -> "ax":
        return self.axes[idx]

