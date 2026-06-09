import inspect
import json
import pickle
from functools import wraps
from typing import Any, Dict, Iterable, NoReturn

import matplotlib.pyplot as plt

from .config import style_cfg


def load(filename: str) -> Dict:
    with open(filename, encoding="utf-8") as j:
        data = json.load(j)
    return data


def export_pickle(data: Any, file_: str) -> NoReturn:
    fh = None
    try:
        fh = open(file_, "wb")
        pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)
    except (EnvironmentError, pickle.PicklingError) as err:
        ExportError_ = type("ExportError", (Exception,), dict())
        raise ExportError_(f"Export Error: {err}")
    finally:
        if fh is not None:
            fh.close()


def import_pickle(file_: str) -> Any:
    fh = None
    try:
        fh = open(file_, "rb")
        return pickle.load(fh)
    except (EnvironmentError, pickle.UnpicklingError) as err:
        raise ImportError(f"Import Error: {err}")
    finally:
        if fh is not None:
            fh.close()


def inherit_from_matplotlib(func):
    @wraps(func)
    def wrapper(self, index=(0, 0), **kwargs):
        axes = self[index]
        if isinstance(axes, Iterable):
            return [getattr(ax, func.__name__)(**kwargs) for ax in axes]
        else:
            return getattr(axes, func.__name__)(**kwargs)

    wrapper.__signature__ = inspect.signature(func)
    return wrapper


def get_style(style_type):
    assert style_type is not None, "style should not be None ..."
    if isinstance(style_type, Dict):
        return [style_type]
    try:
        style = style_cfg[style_type]
    except KeyError:
        style = [style_type]
    return style


def style_env(func):
    @wraps(func)
    def wrapper(*args, **new_kwargs):
        style = []
        kwargs = dict(inspect.getfullargspec(func).kwonlydefaults or {})
        kwargs.update(**new_kwargs)
        if isinstance(kwargs["style"], str):
            style += get_style(kwargs["style"])
        else:
            for item in kwargs["style"]:
                style += get_style(item)
        with plt.style.context(style, after_reset=False):
            return func(*args, **kwargs)

    wrapper.__signature__ = inspect.signature(func)
    return wrapper
