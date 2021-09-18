



from typing import List, Tuple, Optional, Dict, Union, Iterable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import mpl_toolkits.axisartist as AA
import os

from .config import cfg, style_cfg
from .utils import getmore, get_style, style_env



class UnitAX:

    def __init__(
        self, axes, gs, anchor: 'UnitAX',
        sharey: bool = True, **kwargs
    ):
        self.axes = axes
        self.gs = gs
        self.anchor = anchor
        self.sharey = sharey
        self.kwargs = kwargs
        self.__ax = None

    @property
    def ax(self):
        if self.__ax is None:
            if not self.sharey or self.anchor is None:
                self.__ax = self.axes.fig.add_subplot(
                    self.gs, **self.kwargs
                )
            else:
                self.__ax = self.axes.fig.add_subplot(
                    self.gs, sharey=self.anchor.ax, **self.kwargs
                )
                plt.setp(self.__ax.get_yticklabels(), visible=False)
        return self.__ax


class FreeAxes:

    def __init__(
        self, fig, shapes: Iterable, 
        titles: Optional[Iterable] = None,
        sharey: bool = True, projection: Optional[str] = None,
    ):

        self.fig = fig
        self.axes = []
        if titles is not None:
            titles = np.array(titles).reshape(shapes)

        grids = fig.add_gridspec(*shapes)
        for i in range(shapes[0]):
            self.axes.append([])
            anchor = UnitAX(self, grids[i, 0], anchor=None, sharey=False, projection=projection)
            self.axes[-1].append(anchor)
            for j in range(1, shapes[1]):
                ax = UnitAX(self, grids[i, j], anchor=anchor, sharey=sharey, projection=projection)
                self.axes[-1].append(ax)

        self.axes = np.array(self.axes)
        self.links = self._get_links(titles)
        self.titles = np.array(list(self.links.keys()))

    def _get_links(self, titles: Optional[Iterable]) -> Dict:
        m, n = self.axes.shape
        names = dict()
        if titles is None:
            for i in range(m):
                for j in range(n):
                    s = "(" + chr(i + 97) + ")"
                    names.update({s:(i, j)})
        else:
            for i in range(m):
                for j in range(n):
                    title = titles[i, j]
                    names.update({title:(i, j)})
        return names

    def set(self, index: Union[Axes, str, Iterable[int], slice, None] = None, **kwargs) -> None:
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
    """
    A simple implement is used to draw some easy figures in my sense. 
    It is actually a rewrite based on matplotlib and seaborn as the former 
    is flexible but difficult to use and the latter is eaiser but not flexible.
    Therefore, I try my best to combine the both to make it easy to draw.
    At least, in my opinion, it's helpful.
    """
    def __init__(
        self, 
        shape: Tuple[int, int], 
        figsize: Tuple[float, float], 
        titles: Optional[Iterable]=None,
        sharey: bool = True,
        projection: Optional[str] = None,
        **kwargs: "other kwargs of plt.subplots"
    ):
        """
        If you are familiar with plt.subplots, you will find most of 
        kwargs can be used here directly except
        titles: a list or tuple including the subtitles for differents axes.
        You can ignore this argument and we will assign (a), (b) ... as a default setting.
        Titles will be useful if you want call a axe by the subtitles or endowing the axes 
        different titles together.
        """
        # the default settings
        plt.style.use(style_cfg.basic)
        for group, params in cfg['rc_params'].items():
            plt.rc(group, **params)

        self.fig = plt.figure(figsize=figsize, **kwargs)
        self.grids = self.fig.add_gridspec(*shape)
        self.axes = FreeAxes(self.fig, shape, titles, sharey, projection=projection)
    
    @property
    def styles(self):
        return plt.style.available + list(style_cfg.keys())
    
    @property
    def rcParams(self):
        return matplotlib.rcParams

    def set(self, index: Union[str, Iterable[int], slice, None] = None, **kwargs) -> None:
        self.axes.set(index=index, **kwargs)

    def set_style(self, style: Union[str, Iterable[str]]):
        styles = []
        if isinstance(style, str):
            styles += get_style(style)
        else:
            for item in style:
                styles += get_style(item)
        plt.style.use(styles)

    def set_scale(self, value: str = 'symlog', index=(0, 0), axis: str = 'y') -> None:
        """
        value: 'log'|'linear'|'symlog'|'logit'
        """
        kwargs = dict()
        kwargs['index'] = index
        kwargs[axis + 'scale'] = value
        return self.set(**kwargs)

    def set_title(self, y: float = .99) -> None:
        self.axes.set_title(y=y)

    def set_ticks(self, values, index=(0, 0), fmt: str = "%2f", axis: str = 'y') -> Dict:
        labels = [fmt%value for value in values]
        kwargs = dict()
        kwargs['index'] = index
        kwargs[axis + 'ticks'] = values
        kwargs[axis + 'ticklabels'] = labels
        return self.set(**kwargs)

    def set_lim(self, lim: Iterable[float], index=(0, 0), axis='y'):
        kwargs = dict()
        kwargs['index'] = index
        kwargs[axis + 'lim'] = lim
        return self.set(**kwargs)

    def set_label(self, label: str, index=(0, 0), axis='y'):
        kwargs = dict()
        kwargs['index'] = index
        kwargs[axis + 'label'] = label
        return self.set(**kwargs)

    
    def ticklabel_format(
        self, style: str = 'sci', scilimits: Iterable[int] = (0, 0),
        index=None, axis: str = 'y', **kwargs
    ):
        """
        When encountering very big numbers such 7000000,
        it will be ugly to display them on the ticks directly.
        By calling ticklabel_format with style=='sci',
        it will become 7 x 10^6.
        style: 'sci'|'scientific'|'plain', 'plain' is 7000000 actually.
        'scilimits': (m, n); Only numbers between 10^m and 10^n will be
                transformed to scientific notation. The default settings (0, 0)
                means all numbers will be transformed.
        axis: 'x'|'y'|'both'
        """
        self.axes.ticklabel_format(index=index, style=style, scilimits=scilimits, axis=axis, **kwargs)


    @getmore("get the facecolor of the Axes") 
    def get_facecolor(self, index=(0, 0)) -> Tuple[float]: ...

    @getmore("return Legend instance or None if no legend")
    def get_legend(self, index=(0, 0)) -> Optional[matplotlib.legend.Legend]: ...

    @getmore("return handles and labels for legend")
    def get_legend_handles_labels(self, index=(0, 0), legend_handler_map=None) -> Tuple[List]: ...
    
    @getmore("return a list of lines contained in Axes")
    def get_lines(self, index=(0, 0)) -> Iterable[matplotlib.lines.Line2D]: ...

    @getmore("return the title of the Axes")
    def get_title(self, index=(0, 0)) -> str: ...

    @getmore("return the XAxis")
    def get_xaxis(self, index=(0, 0)) -> matplotlib.axis.Axis: ...

    @getmore("get the xlabel text string")
    def get_xlabel(self, index=(0, 0)) -> str: ...

    @getmore("return the x-axis view limits")
    def get_xlim(self, index=(0, 0)) -> Tuple[float, float]: ...

    @getmore("return x-scale type")
    def get_xscale(self, index=(0, 0)) -> str: ...

    @getmore("get the xaxis' tick labels.")
    def get_xticklabels(self, index=(0, 0)) -> Iterable[matplotlib.text.Text]: ...

    @getmore("return the xaxis' tick locations in data coordinates")
    def get_xticks(self, index=(0, 0)) -> np.ndarray: ...

    @getmore("return the YAxis")
    def get_yaxis(self, index=(0, 0)) -> matplotlib.axis.Axis: ...
    
    @getmore("get the ylabel text string")
    def get_ylabel(self, index=(0, 0)) -> str: ...

    @getmore("return the y-axis view limits")
    def get_ylim(self, index=(0, 0)) -> Tuple[float, float]: ...

    @getmore("return y-scale type")
    def get_yscale(self, index=(0, 0)) -> str: ...

    @getmore("get the yaxis' tick labels.")
    def get_yticklabels(self, index=(0, 0)) -> Iterable[matplotlib.text.Text]: ...

    @getmore("return the yaxis' tick locations in data coordinates")
    def get_yticks(self, index=(0, 0)) -> np.ndarray: ...

    @style_env
    def inset_axes(
        self, 
        xlims: Iterable[float], ylims: Iterable[float], bounds: Iterable[float],
        *, style: Union[str, Iterable[str]] = None, index=(0, 0),
        patch_params: dict = {'edgecolor':'black', 'linewidth':.7, 'alpha':.5},
        line_params: dict = {'color':'gray', 'linewidth':.5, 'alpha':.7, 'linestyle':'--'}
    ) -> "Axes, Patch, Lines":
        """
        xlims|ylims: (l, r)|(b, t), to determine the retangle region from l to r and from b to t;
        bounds: (x0, y0, width, height), the new inseted ax located in rectangle (x0, y0) to (x0 + width, y0 + height).
                Note that these values are according to the original ax, so they should be in [0, 1].
        style: You must choose the same style as keep consistent with the original ax.
        patch_params: You could specific the patch by passing a style dict;
        line_params: You could specific the line by passing a style dict.
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
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

    def savefig(
        self, filename: str, 
        bbox_inches: str = 'tight', 
        tight_layout: bool = True,
        **kwargs: "other kwargs of plg.savefig"
    ) -> None:
        if tight_layout:
            plt.tight_layout()
        self.fig.savefig(
            filename,
            bbox_inches=bbox_inches,
            **kwargs
        )
    
    def show(self, *args, tight_layout: bool = True, **kwargs):
        if tight_layout:
            plt.tight_layout()
        return plt.show(*args, **kwargs)

    def __getitem__(self, idx: Union[str, Tuple[int]]) -> "ax":
        return self.axes[idx]

