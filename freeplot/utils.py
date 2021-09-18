
from typing import Dict, Iterable, List
import json
import inspect
import matplotlib.pyplot as plt
from .config import style_cfg

def load(filename: str) -> Dict:
    with open(filename, encoding="utf-8") as j:
        data = json.load(j)
    return data


def axis(func):
    def wrapper(*args, **kwargs):
        axis = kwargs[axis]
        results = func(*args, **kwargs)
        newresults = dict()
        for name, value in results.item():
            newresults[axis + results] = value
        return value
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

def reset(set):
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            results[index] = kwargs[index]
            return set(*results)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator

def getmore(doc):
    def decorator(func):
        def wrapper(self, index=(0, 0), **kwargs):
            axes = self[index]
            if isinstance(axes, Iterable):
                return [getattr(ax, func.__name__)(**kwargs) for ax in axes]
            else:
                return getattr(axes, func.__name__)(**kwargs)
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = doc
        return wrapper
    return decorator

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
    def wrapper(*arg, **new_kwargs):
        style = []
        kwargs = inspect.getfullargspec(func).kwonlydefaults
        kwargs.update(**new_kwargs)
        if isinstance(kwargs['style'], str):
            style += get_style(kwargs['style'])
        else:
            for item in kwargs['style']:
                style += get_style(item)
        with plt.style.context(style, after_reset=False):
            results = func(*arg, **kwargs)
        return results
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

