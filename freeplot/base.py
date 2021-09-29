



from typing import Iterable, Tuple, Optional, Dict, Union
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from .unit import UnitPlot
from .utils import style_env

 

class FreePlot(UnitPlot):

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
        sns.heatmap(
            data, ax=ax, 
            annot=annot, fmt=fmt,
            cmap=cmap, linewidth=linewidth,
            **kwargs
        )

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
            other kwargs of ax.plt or sns.lineplot
        """
        ax = self[index]
        if seaborn:
            sns.lineplot(x, y, ax=ax, **kwargs)
        else:
            ax.plot(x, y, **kwargs)
        
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
    
    def imread(self, fname: str) -> np.ndarray:
        """load the image"""
        return plt.imread(fname)

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
    def barplot(
        self, x: str, y: str, hue: str, 
        data: pd.DataFrame, 
        index: Union[Tuple[int], str] = (0, 0), 
        auto_fmt: bool = False, *,
        style: Union[str, Iterable[str]] = 'bar',
        **kwargs: "other kwargs of sns.barplot"
    ) -> None:
        """
        Args:
            x, y, hue: the colname of x, y and hue;
            auto_fmt: adjust the xticklabel if True;
        Kwargs:
            palette: Dict, set the color of each of hue.
            ...
        """
        ax = self[index]
        sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax, **kwargs)
        if auto_fmt:
            self.fig.autofmt_xdate()

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
        ax.violinplot(dataset=y, **kwargs)
        ax.set(
            xticks=range(1, len(y) + 1),
            xticklabels=x
        )

   
        




