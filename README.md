# FreePlot

FreePlot is a Python data visualization library based on Matplotlib. It provides a compact plotting container and common chart helpers for experiment figures and paper-ready visualizations.

## Features

- Grid-like figure container with title-based or index-based Axes access
- Common plotting helpers for line, scatter, bar, heatmap, violin, radar, inset, and 3D surface figures
- Matplotlib-compatible API surface, so lower-level Axes customization remains available
- Documentation-first examples that can run in non-GUI environments

## Installation

FreePlot requires Python 3.9 or newer.

```bash
pip install freeplot
```

For local development:

```bash
pip install -e ".[dev,docs]"
```

## Quick Start

```python
import matplotlib

matplotlib.use("Agg")

import numpy as np

from freeplot import FreePlot

x = np.linspace(0, 2, 20)
y = x**2

fp = FreePlot(shape=(1, 1), figsize=(2.4, 3.2), dpi=160)
fp.lineplot(x, y, label="x^2")
fp.set(xlabel="x", ylabel="y")
fp[0, 0].legend()
fp.savefig("quickstart.png")
```

## Documentation

Documentation lives in [`docs/`](docs/index.rst). Build it locally with:

```bash
sphinx-build -b html docs docs/_build/html
```

## Preview

![FreePlot overview](README.assets/demo.png)

## License

FreePlot is released under the MIT License.
