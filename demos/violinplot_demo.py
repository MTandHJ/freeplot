

import numpy as np
import matplotlib.pyplot as plt
from freeplot.base import FreePlot




fp = FreePlot((1, 1), (5, 5))
# note that each element is a group of data ...
all_data = [np.random.normal(0, std, 100) for std in range(5, 10)]
fp.violinplot(x=None, y=all_data, index=0)

plt.show()
