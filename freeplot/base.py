



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
        auto_fmt: bool = False, *,
        hatch: Optional[Iterable] = None,
        hatch_scale: int = 3,
        style: Union[str, Iterable[str]] = 'bar',
        **kwargs: "other kwargs of sns.barplot"
    ) -> None:
        """
        Args:
            x, y, hue: the colname of x, y and hue;
            auto_fmt: adjust the xticklabel if True;
        Kwargs:
            palette: Dict, set the color of each of hue.
            ci: float or 'sd', If “sd”, skip bootstrapping and draw the standard deviation of the observations. 
                If None, no bootstrapping will be performed, and error bars will not be drawn.
            hatch: ["-", "/", "\\", "x", "+", ".", "*"].
            hatch_scale: hatch * hatch_scale.
            ...
        """
        ax = self[index]
        sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax, **kwargs)
        if auto_fmt:
            self.fig.autofmt_xdate()
        if hatch:
            for pattern, bars in zip(hatch, self.get_container(index=index)):
                for bar in bars:
                    bar.set_hatch(pattern * hatch_scale)

    @style_env
    def contourf(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
        levels: Optional[Union[int, np.ndarray]] = 5, cbar: bool = True,
        index: Union[Tuple[int], str] = (0, 0), *,
        style: Union[str, Iterable[str]] = [],
        origin: Optional[str] = 'lower', cmap = plt.cm.bone,
        **kwargs: "other kwargs of ax.contourf"
    ):
        ax = self[index]
        cs =  ax.contourf(X, Y, Z, levels, cmap=cmap, origin=origin, **kwargs)
        if cbar:
            self.fig.colorbar(cs)
        return cs

    @style_env
    def heatmap(
        self, data: pd.DataFrame, 
        index: Union[Tuple[int], str] = (0, 0), 
        annot: bool = True, 
        fmt: str = ".4f",
        cmap: str = 'GnBu', 
        linewidth: float = .5, *,
        style: Union[str, Iterable[str]] = 'heatmap',
        **kwargs: "other kwargs of sns.heatmap"
    ) -> None:
        """
        Args:
            data: M x N dataframe.
            cmap: GnBu, Oranges are recommanded.
            annot: annotation.
        Kwargs:
            fmt: the format for annotation.
            kwargs:
                cbar: bool
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
        **kwargs: "other kwargs of ax.imshow"
    ) -> None:
        """
        Args:
            show_ticks: show the ticks if True
        Kwargs: other kwargs of ax.imshow
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
        index: Union[Tuple[int], str] = (0, 0), 
        seaborn: bool = False, *,
        style: Union[str, Iterable[str]] = 'line',
        **kwargs: "other kwargs of ax.plot or sns.lineplot"
    ) -> None:
        """
        Args:
            x, y: Iterable;
            seaborn: bool, use sns.lineplot to plot if True
        Kwargs:
            marker: '' for no markers
            other kwargs of ax.plt or sns.lineplot
        """
        ax = self[index]
        if seaborn:
            return sns.lineplot(x, y, ax=ax, **kwargs)
        else:
            return ax.plot(x, y, **kwargs)
        
    @style_env
    def scatterplot(
        self, x: np.ndarray, y: np.ndarray, 
        index: Union[Tuple[int], str] = (0, 0), 
        seaborn: bool = False, *,
        style: Union[str, Iterable[str]] = 'scatter',
        **kwargs: "other kwargs of ax.scatter or sns.scatterplot"
    ) -> None:
        """
        Args:
            x, y: Iterable;
            seaborn: bool, use sns.scatterplot to plot if True;
        Kwargs:
            other kwargs of ax.scatter or sns.scatterplot
        """
        ax = self[index]
        if seaborn:
            sns.scatterplot(x, y, ax=ax, **kwargs)
        else:
            ax.scatter(x, y, **kwargs)


    @style_env
    def violinplot(
        self, y: Iterable, x: Optional[Iterable[str]] = None,
        index: Union[Tuple[int], str] = (0, 0), *,
        style: Union[str, Iterable[str]] = 'violin',
        **kwargs: "other kwargs of ax.violinplot"
    ) -> None:
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








