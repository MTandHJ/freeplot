

"""
heatmap demo
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys
import seaborn as sns
from freeplot.base import FreePlot


os.chdir(sys.path[0])


titles = ("S", "h", "a", "n")

row_labels = ('c', 'u', 't', 'e')
col_labels = ('l', 'r', 'i', 'g')

# shape: 1, 4; figsize: 9, 2
fp = FreePlot((1, 4), (9, 2), titles=titles, dpi=100, sharey=True)

for title in titles:
    data = np.random.rand(4, 4)
    df = pd.DataFrame(data, index=col_labels, columns=row_labels)
    fp.heatmap(df, index=title, annot=True, fmt=".4f", cbar=False, linewidth=0.5)

fp.set(Xlabel="X")
fp.set_label('Y', index=(0, 0), axis='y')
# fp.savefig("heatmap_demo.pdf", format="pdf", tight_layout=False)
plt.show()