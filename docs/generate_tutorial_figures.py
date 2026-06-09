"""Generate tutorial figures for the documentation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from freeplot import FreePlot
from freeplot.zoo import pos_radar, pre_radar

OUTPUT_DIR = Path(__file__).parent / "_static" / "img" / "tutorials"


def save(fp: FreePlot, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp.savefig(OUTPUT_DIR / name)


def line() -> None:
    x = np.linspace(0, 2, 80)
    fp = FreePlot()
    fp.lineplot(x, x**0.5, label="sqrt")
    fp.lineplot(x, x**2, label="square", marker="")
    fp[0, 0].legend()
    save(fp, "line.png")


def scatter() -> None:
    rng = np.random.default_rng(0)
    fp = FreePlot()
    fp.scatterplot(rng.normal(size=160), rng.normal(size=160), edgecolors="none", alpha=0.75)
    fp.set_label("x", axis="x")
    fp.set_label("y", axis="y")
    save(fp, "scatter.png")


def bar() -> None:
    data = pd.DataFrame(
        {
            "name": ["One", "Two", "Three"] * 2,
            "value": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )
    fp = FreePlot()
    fp.barplot(x="name", y="value", hue="group", data=data, hatch=["", "/"], errorbar=None)
    save(fp, "bar.png")


def histogram() -> None:
    rng = np.random.default_rng(1)
    fp = FreePlot()
    fp.histplot(rng.normal(size=300), num_bins=24, density=True)
    fp.set_label("density")
    save(fp, "histogram.png")


def heatmap() -> None:
    data = pd.DataFrame(np.arange(16).reshape(4, 4), columns=list("ABCD"))
    fp = FreePlot()
    fp.heatmap(data, annot=True, fmt=".0f", cbar=False)
    save(fp, "heatmap.png")


def image() -> None:
    x = np.linspace(0, 1, 80)
    img = np.outer(np.sin(np.pi * x), np.cos(np.pi * x))
    fp = FreePlot()
    fp.imageplot(img)
    save(fp, "image.png")


def stack() -> None:
    x = np.arange(6)
    y = np.vstack([np.ones(6), np.arange(1, 7), np.linspace(2, 4, 6)])
    fp = FreePlot()
    fp.stackplot(x, y, labels=["base", "growth", "trend"])
    fp[0, 0].legend()
    save(fp, "stack.png")


def violin() -> None:
    rng = np.random.default_rng(2)
    values = [rng.normal(0, std, 120) for std in (1, 2, 3)]
    fp = FreePlot()
    fp.violinplot(values, x=["std-1", "std-2", "std-3"])
    save(fp, "violin.png")


def contour() -> None:
    x = np.arange(-3, 3, 0.15)
    y = np.arange(-3, 3, 0.15)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    fp = FreePlot()
    fp.contourf(X, Y, Z, levels=12)
    save(fp, "contour.png")


def inset() -> None:
    x = np.linspace(0, 4, 160)
    y = np.sin(x)
    fp = FreePlot()
    fp.lineplot(x, y, label="sin")
    axins, _patch, _lines = fp.inset_axes(
        xlims=(1.2, 1.8),
        ylims=(0.9, 1.05),
        bounds=(0.55, 0.12, 0.35, 0.35),
        style="line",
    )
    fp.lineplot(x, y, index=axins, marker="")
    save(fp, "inset.png")


def surface() -> None:
    x = np.arange(-4, 4, 0.25)
    y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    fp = FreePlot(projection="3d")
    fp.surfaceplot(X, Y, Z, linewidth=0)
    save(fp, "surface.png")


def radar() -> None:
    labels = np.array(["A", "B", "C", "D", "E"])
    theta = pre_radar(len(labels), frame="polygon")
    data = {
        "left": np.array([0.2, 0.7, 0.5, 0.8, 0.6]),
        "right": np.array([0.8, 0.4, 0.6, 0.3, 0.9]),
    }
    fp = FreePlot(projection="radar")
    pos_radar(data, labels, fp, theta=theta)
    fp[0, 0].legend()
    save(fp, "radar.png")


def main() -> None:
    for plot in (
        line,
        scatter,
        bar,
        histogram,
        heatmap,
        image,
        stack,
        violin,
        contour,
        inset,
        surface,
        radar,
    ):
        plot()


if __name__ == "__main__":
    main()
