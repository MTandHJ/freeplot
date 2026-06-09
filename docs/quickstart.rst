快速开始
========

下面的示例创建一个单图容器, 绘制折线图并保存图片。

.. code-block:: python

   import matplotlib

   matplotlib.use("Agg")

   import numpy as np

   from freeplot import FreePlot

   x = np.linspace(0, 2, 20)
   y = x**2

   fp = FreePlot(shape=(1, 1), figsize=(2.4, 3.2), dpi=160)
   fp.lineplot(x, y, label="x^2", marker="")
   fp.set(xlabel="x", ylabel="y")
   fp[0, 0].legend()
   fp.savefig("quickstart.png")

常用操作
--------

.. code-block:: python

   fp = FreePlot(shape=(2, 2), titles=("Line", "Scatter", "Bar", "Heatmap"))

   fp[0, 0]       # 通过位置访问 Axes
   fp["Line"]     # 通过标题访问 Axes

   fp.set_label("X", axis="x", index=(0, 0))
   fp.set_label("Y", axis="y", index=(0, 0))
   fp.set_title(y=0.98)
