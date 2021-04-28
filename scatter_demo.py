


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from freeplot.base import FreePlot
from scipy.stats import multivariate_normal

normal = multivariate_normal
titles = ("S", "h", "a", "n")
labels = ("sin", "cos", "x")

# shape: 1, 4; figsize: 9.5, 2
fp = FreePlot((1, 4), (9.5, 2), titles=titles, dpi=100, sharey=True)

nums = 100
means = (
    (0, 0),
    (5, 5),
    (-5, -5)
)

cov = (
    2,
    1,
    1
)

for title in titles:
    for i, mean in enumerate(means):
        data = normal.rvs(mean, cov[i], size=nums)
        fp.scatterplot(data[:, 0], data[:, 1], index=title, label=labels[i])
        
fp.set_title(y=1.)
fp.set_label("Y", index=0, axis='y')
fp.set_label("X", index=None, axis='x')
fp[0].legend()
# fp.savefig("scatter_demo.pdf", format="pdf", tight_layout=False)
plt.show()



            









