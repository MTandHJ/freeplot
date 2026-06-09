3D 曲面图
=========

`surfaceplot` 用于在 3D 坐标系中绘制曲面。

.. code-block:: python

   import numpy as np
   from freeplot import FreePlot

   x = np.arange(-4, 4, 0.25)
   y = np.arange(-4, 4, 0.25)
   X, Y = np.meshgrid(x, y)
   Z = np.sin(np.sqrt(X**2 + Y**2))

   fp = FreePlot(projection="3d")
   fp.surfaceplot(X, Y, Z, linewidth=0)
   fp.savefig("surface.png")

.. image:: ../_static/img/tutorials/surface.png
   :align: center
   :alt: 3D surface plot
