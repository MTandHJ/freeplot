import matplotlib

matplotlib.use("Agg")

import numpy as np

from freeplot import FreePlot


def test_quickstart_example_saves_figure(tmp_path) -> None:
    x = np.linspace(0, 2, 20)
    y = x**2
    output = tmp_path / "quickstart.png"

    fp = FreePlot(shape=(1, 1), figsize=(2.4, 3.2), dpi=160)
    fp.lineplot(x, y, label="x^2")
    fp.set(xlabel="x", ylabel="y")
    fp[0, 0].legend()
    fp.savefig(output)

    assert output.exists()
    assert output.stat().st_size > 0
