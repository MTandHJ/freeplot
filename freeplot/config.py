



from cycler import cycler


class Config(dict):
    '''
    >>> cfg = Config({1:2}, a=3)
    Traceback (most recent call last):
    ...
    TypeError: attribute name must be string, not 'int'
    >>> cfg = Config(a=1, b=2)
    >>> cfg.a
    1
    >>> cfg['a']
    1
    >>> cfg['c'] = 3
    >>> cfg.c
    3
    >>> cfg.d = 4
    >>> cfg['d']
    Traceback (most recent call last):
    ...
    KeyError: 'd'
    '''
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        for name, attr in self.items():
            self.__setattr__(name, attr)

    def __setitem__(self, key, value):
        super(Config, self).__setitem__(key, value)
        self.__setattr__(key, value)



COLORS = (
    "#2D2C40",
    "#B3002A",
    "#0050C2",
    "#006058",
    "#3A4E8C",
    "#D99F6C",
    "#BF3F34",
)

MARKERS = (
    'o',
    '^',
    's',
    'D',
    'p',
    '*',
    '+',
)

cfg = Config()


#rc

_axes = {
    "prop_cycle": cycler(marker=MARKERS, color=COLORS),
    "titlesize": 11,
    "labelsize": 11,
    "facecolor": 'white',
    "grid": False
}

_font = {
        "family": ["serif"],
        "weight": "normal",
        "size": 7
    }

_lines = {
    "linewidth": 1.,
    # "marker": None,
    "markersize": 5,
    "markeredgewidth": .5,
    "markerfacecolor": "auto",
    "markeredgecolor": "white",
}

_markers = {
    # "fillstyle": "none"
}


_legend = {
    'borderaxespad': 0.3,
    'borderpad': 0.2,
    'columnspacing': 0.1,
    'edgecolor': '0.8',
    'facecolor': 'inherit',
    'fancybox': True,
    'fontsize': 10,
    'framealpha': 0.5,
    'frameon': False,
    'handleheight': 0.7,
    'handlelength': 2.0,
    'handletextpad': 0.8,
    'labelspacing': 0.5,
    'loc': 'best',
    'markerscale': 1.0,
    'numpoints': 1,
    'scatterpoints': 1,
    'shadow': False,
    'title_fontsize': None,
}

_xtick = {
    'alignment': 'center',
    'bottom': True,
    'color': 'black',
    'direction': 'out',
    'labelbottom': True,
    'labelsize': 11,
    'labeltop': False,
    'major.bottom': True,
    'major.pad': 3.5,
    'major.size': 3.5,
    'major.top': True,
    'major.width': 0.5,
    'minor.bottom': True,
    'minor.pad': 3.4,
    'minor.size': 2.0,
    'minor.top': True,
    'minor.visible': False,
    'minor.width': 0.4,
    'top': False,
}

_ytick = {
    'alignment': 'center_baseline',
    'color': 'black',
    'direction': 'out',
    'labelleft': True,
    'labelright': False,
    'labelsize': 11,
    'left': True,
    'major.left': True,
    'major.pad': 3.5,
    'major.right': True,
    'major.size': 3.5,
    'major.width': 0.5,
    'minor.left': True,
    'minor.pad': 3.4,
    'minor.right': True,
    'minor.size': 2.0,
    'minor.visible': False,
    'minor.width': 0.4,
    'right': False
}

cfg['rc_params'] = Config(
    axes=_axes,
    font=_font,
    lines=_lines,
    markers=_markers,
    legend=_legend,
    xtick=_xtick,
    ytick=_ytick
)

style_cfg = Config()

style_cfg['basic'] = ["science"]  # color style: bright, vibrant, muted, high-contrast, light, high-vis, retro
style_cfg['line'] = [] 
style_cfg['scatter'] = []
style_cfg['heatmap'] = ["seaborn-darkgrid", {"axes.facecolor":".9"}]
style_cfg['image'] = ["bright"]
style_cfg['bar'] = []
style_cfg['violin'] = ["high-vis", "seaborn-whitegrid"]
style_cfg['surface'] = [{"axes.facecolor":".3"}]

# zoo
style_cfg['radar'] = []

    

