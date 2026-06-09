import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.collections import PolyCollection
from matplotlib.contour import QuadContourSet
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from freeplot import FreePlot
from freeplot.base import FreePatches
from freeplot.zoo import pos_radar, pre_radar


def assert_saved(fp: FreePlot, tmp_path, name: str) -> None:
    output = tmp_path / name
    fp.savefig(output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_lineplot(tmp_path) -> None:
    x = np.linspace(0, 2, 20)
    fp = FreePlot()

    lines = fp.lineplot(x, x**2, label="square")

    assert isinstance(lines[0], Line2D)
    assert len(fp.get_lines()) == 1
    assert_saved(fp, tmp_path, "line.png")


def test_scatterplot(tmp_path) -> None:
    rng = np.random.default_rng(0)
    fp = FreePlot()

    points = fp.scatterplot(rng.normal(size=20), rng.normal(size=20), edgecolors="none")

    assert points.get_offsets().shape[0] == 20
    assert_saved(fp, tmp_path, "scatter.png")


def test_barplot(tmp_path) -> None:
    data = pd.DataFrame(
        {
            "name": ["One", "Two", "Three"] * 2,
            "value": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )
    fp = FreePlot()

    handles, labels = fp.barplot(
        x="name", y="value", hue="group", data=data, hatch=["", "/"], errorbar=None
    )

    assert labels == ["A", "B"]
    assert handles
    assert fp.get_containers()
    assert_saved(fp, tmp_path, "bar.png")


def test_contourf(tmp_path) -> None:
    x = np.arange(-2, 2, 0.25)
    y = np.arange(-2, 2, 0.25)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    fp = FreePlot()

    contour = fp.contourf(X, Y, Z, levels=5, cbar=False)

    assert isinstance(contour, QuadContourSet)
    assert_saved(fp, tmp_path, "contour.png")


def test_histplot(tmp_path) -> None:
    rng = np.random.default_rng(1)
    fp = FreePlot()

    fp.histplot(rng.normal(size=100), num_bins=10, density=True)

    assert fp.get_patches()
    assert_saved(fp, tmp_path, "hist.png")


def test_heatmap(tmp_path) -> None:
    data = pd.DataFrame(np.arange(9).reshape(3, 3), columns=list("abc"))
    fp = FreePlot()

    ax = fp.heatmap(data, annot=True, fmt=".0f", cbar=False)

    assert ax is fp[0, 0]
    assert_saved(fp, tmp_path, "heatmap.png")


def test_imageplot_grayscale_and_rgb(tmp_path) -> None:
    fp = FreePlot(shape=(1, 2), sharey=False)

    fp.imageplot(np.arange(9).reshape(3, 3), index=(0, 0))
    fp.imageplot(np.ones((3, 3, 3)), index=(0, 1), show_ticks=True)

    assert isinstance(fp[0, 0].images[0], AxesImage)
    assert isinstance(fp[0, 1].images[0], AxesImage)
    assert_saved(fp, tmp_path, "image.png")


def test_stackplot(tmp_path) -> None:
    x = np.arange(5)
    y = np.vstack([np.ones(5), np.arange(1, 6)])
    fp = FreePlot()

    stacks = fp.stackplot(x, y)

    assert all(isinstance(item, PolyCollection) for item in stacks)
    assert_saved(fp, tmp_path, "stack.png")


def test_surfaceplot(tmp_path) -> None:
    x = np.arange(-2, 2, 0.5)
    y = np.arange(-2, 2, 0.5)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    fp = FreePlot(projection="3d")

    surface = fp.surfaceplot(X, Y, Z, linewidth=0)

    assert surface is not None
    assert_saved(fp, tmp_path, "surface.png")


def test_violinplot(tmp_path) -> None:
    rng = np.random.default_rng(2)
    values = [rng.normal(0, std, 50) for std in (1, 2, 3)]
    fp = FreePlot()

    obj = fp.violinplot(values, x=["one", "two", "three"])

    assert obj["bodies"]
    assert_saved(fp, tmp_path, "violin.png")


def test_add_patch_and_free_patches(tmp_path) -> None:
    factory = FreePatches(alpha=0.3, fill=True)
    fp = FreePlot()

    circle = factory.Circle(0.5, 0.5, 0.2, color="tab:blue")
    polygon = factory.Polygon(
        np.array([0.1, 0.2, 0.3]),
        np.array([0.1, 0.3, 0.1]),
        color="tab:orange",
    )
    added = fp.add_patch(circle)
    added_polygon = fp.add_patch(polygon)
    direct = fp.add_patch(Circle((0.2, 0.2), 0.1, fill=False))

    assert isinstance(added, patches.Circle)
    assert isinstance(added_polygon, patches.Polygon)
    assert isinstance(direct, patches.Circle)
    assert len(fp.get_patches()) == 3
    assert_saved(fp, tmp_path, "patch.png")


def test_radar_helpers(tmp_path) -> None:
    labels = np.array(["A", "B", "C", "D"])
    theta = pre_radar(len(labels), frame="polygon")
    fp = FreePlot(projection="radar")
    data = {
        "left": np.array([0.2, 0.7, 0.5, 0.8]),
        "right": np.array([0.8, 0.4, 0.6, 0.3]),
    }

    pos_radar(data, labels, fp, theta=theta)

    assert len(fp.get_lines()) == 2
    assert_saved(fp, tmp_path, "radar.png")
