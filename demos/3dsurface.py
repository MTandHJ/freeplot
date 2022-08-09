



import numpy as np
import matplotlib.pyplot as plt
from freeplot.base import FreePlot


X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

fp = FreePlot(projection='3d', dpi=300)
fp.surfaceplot(X, Y, Z, cmap=plt.cm.coolwarm, antialiased=False, linewidth=0)
fp.set_label(r"$x$", axis='x')
fp.set_label(r"$y$", axis='y')
fp.set_label(r"$z$", axis='z')
fp.show()