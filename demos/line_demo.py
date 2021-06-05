


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from freeplot.base import FreePlot


titles = ("S", "h", "a", "n")
labels = ("sin", "cos", "x")
fp = FreePlot((1, 4), (9.5, 2), titles=titles, dpi=100, sharey=True)

nums = 20
x = np.linspace(-10, 10, nums)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = x
ys = (y1, y2, y3)

for title in titles:
    for i, y in enumerate(ys):
        y = y + np.random.randn(nums)
        fp.lineplot(x, y, index=title, label=labels[i])

fp.set_title(y=1.)
fp.set_label("y", axis='y')
fp.set(Xlabel="x")
fp[0, 0].legend()
# fp.savefig("line_demo.pdf", format="pdf", tight_layout=False)
plt.show()



