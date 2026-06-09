import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.axes import Axes

from freeplot import FreePlot


def test_axes_access_by_index_title_and_axes_instance() -> None:
    fp = FreePlot(shape=(1, 2), titles=("left", "right"), sharey=False)

    left = fp[0, 0]
    right = fp["right"]

    assert isinstance(left, Axes)
    assert isinstance(right, Axes)
    assert fp[left] is left


def test_setters_and_getters_update_axes_state() -> None:
    fp = FreePlot(titles=("main",))

    fp.set_label("x-label", axis="x")
    fp.set_label("y-label", axis="y")
    fp.set_lim((0, 10), axis="x")
    fp.set_lim((-1, 1), axis="y")
    fp.set_scale("linear", axis="y")
    fp.set_ticks([0, 1, 2], axis="x")
    fp.set_text(0.5, 0.5, "note")
    fp.set_arrow(0.1, 0.1, 0.2, 0.2, width=0.01, head_width=0.05)
    fp.set_title(y=0.95)

    assert fp.get_xlabel() == "x-label"
    assert fp.get_ylabel() == "y-label"
    assert fp.get_xlim() == (0.0, 10.0)
    assert fp.get_ylim() == (-1.0, 1.0)
    assert fp.get_yscale() == "linear"
    assert len(fp.get_xticks()) == 3
    assert fp.get_title() == "main"
    assert fp.get_xaxis() is fp[0, 0].xaxis
    assert fp.get_yaxis() is fp[0, 0].yaxis


def test_style_color_font_and_rcparams_properties() -> None:
    fp = FreePlot()

    fp.colors = "bright"
    fp.set_style("default")
    fp.set_font(size=9)

    assert len(fp.colors) > 0
    assert "no-latex" in fp.styles
    assert fp.rcParams["font.size"] == 9


def test_fill_between_legend_inset_and_savefig(tmp_path) -> None:
    x = np.linspace(0, 1, 20)
    fp = FreePlot()

    fp.lineplot(x, x, label="line")
    fill = fp.fill_between(x, x - 0.1, x + 0.1)
    legend = fp.set_figure_legend(0.5, 0.95, ncol=1)
    axins, patch, lines = fp.inset_axes(
        xlims=(0.2, 0.4), ylims=(0.2, 0.4), bounds=(0.55, 0.1, 0.35, 0.35), style="line"
    )
    fp.lineplot(x, x, index=axins)
    fp.subplots_adjust(wspace=0.1, hspace=0.1)
    fp.ticklabel_format(style="plain")

    output = tmp_path / "unit.png"
    fp.savefig(output, tight_layout=False)

    assert fill is not None
    assert legend is not None
    assert patch is not None
    assert lines
    assert output.exists()
    assert output.stat().st_size > 0


def test_imread_loads_saved_image(tmp_path) -> None:
    fp = FreePlot()
    fp.lineplot([0, 1], [0, 1])
    output = tmp_path / "read.png"
    fp.savefig(output)

    image = FreePlot.imread(output)

    assert image.ndim in (2, 3)
    assert image.size > 0
