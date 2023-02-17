

"""
bar demo
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import axisartist
import json
import os
import sys
import seaborn as sns
from freeplot.base import FreePlot



A = [1., 2., 3.]
B = [2., 3., 4.]
T = ['One', 'Two', 'Three'] * 2
Hue = ['A'] * len(A) + ['B'] * len(B)

data = pd.DataFrame(
    {
        "T": T,
        "val": A + B,
        "category": Hue
    }
)

# shape: 1, 1; figsize: 2.2, 2
fp = FreePlot((1, 1), titles=("Bar Demo",), dpi=200)
fp.barplot(x='T', y='val', hue='category', data=data, index=(0, 0), auto_fmt=True)

fp.set(xlabel='X')
fp.set_label('Y', index=(0, 0), axis='y')
fp[0, 0].legend(ncol=2)
# fp.savefig("heatmap_demo.pdf", format="pdf", tight_layout=False)
plt.show()