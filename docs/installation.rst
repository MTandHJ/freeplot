安装
====

环境要求
--------

FreePlot 要求 Python 3.9 或更新版本。

从 PyPI 安装
------------

.. code-block:: bash

   pip install freeplot

从源码开发安装
--------------

.. code-block:: bash

   pip install -e ".[dev,docs]"

运行验证
--------

.. code-block:: bash

   python -c "import freeplot"
   pytest
   ruff check .
   sphinx-build -b html docs docs/_build/html
