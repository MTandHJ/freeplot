"""Generate tutorial figures for the documentation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freeplot import FreePlot
from freeplot.zoo import pos_radar, pre_radar

IMG_DIR = Path(__file__).parent / "_static" / "img"
OUTPUT_DIR = IMG_DIR / "tutorials"


def save(fp: FreePlot, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp.savefig(OUTPUT_DIR / name)


def line() -> None:
    x = np.linspace(0, 2, 80)
    fp = FreePlot()
    fp.lineplot(x, x**0.5, label="sqrt", marker="")
    fp.lineplot(x, x**2, label="square", marker="")
    fp[0, 0].legend(frameon=False)
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
    fp[0, 0].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
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
    fp.lineplot(x, y, label="sin", marker="")
    axins, _patch, _lines = fp.inset_axes(
        xlims=(1.2, 1.8),
        ylims=(0.9, 1.05),
        bounds=(0.55, 0.12, 0.35, 0.35),
        style="line",
    )
    fp.lineplot(x, y, index=axins, marker="")
    axins.set_xticks([1.3, 1.7])
    axins.tick_params(labelsize=6, labelbottom=False)
    save(fp, "inset.png")


def surface() -> None:
    x = np.arange(-4, 4, 0.25)
    y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    fp = FreePlot(figsize=(2.1, 2.1), projection="3d")
    fp.surfaceplot(X, Y, Z, linewidth=0)
    fp[0, 0].view_init(elev=28, azim=-55)
    fp[0, 0].set_xticks([-3, 0, 3])
    fp[0, 0].set_yticks([-3, 0, 3])
    fp[0, 0].set_zticks([-1, 0, 1])
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
    fp[0, 0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    save(fp, "radar.png")


def overview() -> None:
    plots = [
        ("Line", "line.png"),
        ("Scatter", "scatter.png"),
        ("Bar", "bar.png"),
        ("Histogram", "histogram.png"),
        ("Heatmap", "heatmap.png"),
        ("Image", "image.png"),
        ("Stack", "stack.png"),
        ("Violin", "violin.png"),
        ("Contour", "contour.png"),
        ("Inset", "inset.png"),
        ("Surface", "surface.png"),
        ("Radar", "radar.png"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(8.4, 6.0), dpi=220)
    for ax, (title, filename) in zip(axes.ravel(), plots):
        ax.imshow(plt.imread(OUTPUT_DIR / filename))
        ax.set_title(title, fontsize=8)
        ax.set_axis_off()
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.93, wspace=0.08, hspace=0.24)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG_DIR / "overview.png", bbox_inches="tight")
    plt.close(fig)


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
    overview()


if __name__ == "__main__":
    main()
