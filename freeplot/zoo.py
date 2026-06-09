from typing import Dict, Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .base import FreePlot
from .utils import style_env

__all__ = ["pos_radar", "pre_radar", "roc_curve", "tsne"]


def tsne(
    features: np.ndarray,
    labels: np.ndarray,
    fp: FreePlot,
    index: Union[Tuple[int], str] = (0, 0),
    fontsize: Union[int, str] = "large",
    annotate: bool = False,
    style: Union[str, Iterable[str]] = "bright",
    **kwargs,
) -> None:
    r"""Plot t-SNE embeddings.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix with shape `(n_samples, n_features)`.
    labels : np.ndarray
        Labels with shape `(n_samples,)`.
    fp : FreePlot
        Plot container.
    index : tuple[int, int] or str, default=(0, 0)
        Target axes index or title.
    fontsize : int or str, default="large"
        Annotation font size.
    annotate : bool, default=False
        Whether to annotate each label group.
    style : str or iterable of str, default="bright"
        Style names resolved by `style_env`.
    **kwargs
        Additional keyword arguments passed to `FreePlot.scatterplot`.
    """
    from sklearn.manifold import TSNE

    data_embedded = TSNE(n_components=2, learning_rate=10, n_iter=1000).fit_transform(features)
    fp[index].set_xticks([])
    fp[index].set_yticks([])
    data = pd.DataFrame({"x": data_embedded[:, 0], "y": data_embedded[:, 1], "label": labels})
    for label in np.unique(labels):
        event = data.loc[data["label"] == label]
        x = event["x"]
        y = event["y"]
        if annotate:
            x_mean = x.median()
            y_mean = y.median()
            plt.text(x_mean, y_mean, label, fontsize=fontsize)
        fp.scatterplot(x, y, index, label=label, s=1.5, edgecolors="none", style=style, **kwargs)
    sns.despine(left=True, bottom=True)


def roc_curve(
    y_pred: np.ndarray,
    y_labels: np.ndarray,
    fp: FreePlot,
    index: Union[Tuple[int], str] = (0, 0),
    name: Optional[str] = None,
    estimator_name: Optional[str] = None,
    style: Union[str, Iterable[str]] = "whitegrid",
    dict_: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""Plot an ROC curve.

    Parameters
    ----------
    y_pred : np.ndarray
        Prediction scores.
    y_labels : np.ndarray
        Ground-truth binary labels.
    fp : FreePlot
        Plot container.
    index : tuple[int, int] or str, default=(0, 0)
        Target axes index or title.
    name : str, optional
        Display name for the plotted curve.
    estimator_name : str, optional
        Estimator name shown in the display.
    style : str or iterable of str, default="whitegrid"
        Seaborn axes style.
    dict_ : dict, optional
        Style override dictionary passed to `seaborn.axes_style`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        True positive rates, false positive rates, and ROC-AUC score.
    """
    from sklearn import metrics

    fpr, tpr, _thresholds = metrics.roc_curve(y_labels, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=estimator_name
    )
    with sns.axes_style(style, dict_):
        display.plot(fp[index], name)
    return tpr, fpr, roc_auc


@style_env
def pre_radar(
    num_vars: int, frame: str = "circle", *, style: Union[str, Iterable[str]] = "radar"
) -> np.ndarray:
    r"""Register a radar projection and return angular coordinates.

    Parameters
    ----------
    num_vars : int
        Number of variables around the radar chart.
    frame : {"circle", "polygon"}, default="circle"
        Radar frame shape.
    style : str or iterable of str, default="radar"
        Style names resolved by `style_env`.

    Returns
    -------
    np.ndarray
        Angular coordinates.
    """
    from matplotlib.patches import Circle, RegularPolygon
    from matplotlib.path import Path
    from matplotlib.projections import register_projection
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.spines import Spine
    from matplotlib.transforms import Affine2D

    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = "radar"
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self, spine_type="circle", path=Path.unit_regular_polygon(num_vars)
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


@style_env
def pos_radar(
    data: Dict,
    labels: np.ndarray,
    fp: FreePlot,
    theta: Optional[np.ndarray] = None,
    index: Union[Tuple[int], str] = (0, 0),
    *,
    style: Union[str, Iterable[str]] = "radar",
    alpha: float = 0.5,
) -> None:
    r"""Draw radar data on a registered radar projection.

    Parameters
    ----------
    data : dict
        Mapping from series name to values.
    labels : np.ndarray
        Variable labels.
    fp : FreePlot
        Plot container created with `projection="radar"`.
    theta : np.ndarray, optional
        Angular coordinates returned by `pre_radar`.
    index : tuple[int, int] or str, default=(0, 0)
        Target axes index or title.
    style : str or iterable of str, default="radar"
        Style names resolved by `style_env`.
    alpha : float, default=0.5
        Fill opacity.
    """
    fp.fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False) if theta is None else theta
    ax = fp[index]
    for key, value in data.items():
        ax.plot(theta, value, marker="")
        ax.fill(theta, value, alpha=alpha, label=key)
    ax.set_varlabels(labels)
