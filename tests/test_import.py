import freeplot
from freeplot import FreePlot


def test_import_public_api() -> None:
    assert FreePlot is not None
    assert isinstance(freeplot.__version__, str)
    assert freeplot.__version__
