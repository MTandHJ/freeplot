图像
====

`imageplot` 用于显示灰度图或 RGB 图像。

.. code-block:: python

   import numpy as np
   from freeplot import FreePlot

   x = np.linspace(0, 1, 80)
   img = np.outer(np.sin(np.pi * x), np.cos(np.pi * x))

   fp = FreePlot()
   fp.imageplot(img)
   fp.savefig("image.png")

.. image:: ../_static/img/tutorials/image.png
   :align: center
   :alt: image plot
