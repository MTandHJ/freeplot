# FreePlot 项目规整与文档建设验证设计

## 验证目标

在实施前明确验证方式, 保证项目规整不会破坏基础安装、导入、绘图、文档构建和代码风格检查能力。

本阶段验证不追求完整绘图行为覆盖, 重点建立可重复执行的最小回归基线。

## 目标与非目标

### 目标

- 验证现代构建配置可安装。
- 验证 `freeplot` 包可导入。
- 验证核心绘图容器可创建并保存图片。
- 验证文档中的最小示例可在无 GUI 环境运行。
- 验证 Ruff 检查与格式化流程可执行。
- 验证 Sphinx 文档可本地构建。
- 验证 README 与 docs 的关键说明一致。

### 非目标

- 不做像素级图像一致性测试。
- 不覆盖所有绘图方法组合。
- 不验证交互式 Matplotlib 后端。
- 不验证 PyPI 发布流程。
- 不引入 GitHub Actions 作为本阶段强制验收项。
- 不维护独立 `demos/` 目录的运行测试。

## 用户场景验证

### 新用户安装

场景: 用户从源码安装并尝试导入。

验证方式:

```bash
pip install -e ".[dev,docs]"
python -c "import freeplot"
```

验收标准:

- editable install 成功。
- `import freeplot` 无异常。

### 用户快速绘图

场景: 用户照着 quickstart 创建一张图并保存。

验证方式:

- 在 pytest 中使用 Matplotlib `Agg` 后端。
- 创建 `FreePlot`。
- 调用基础绘图 API。
- 保存到 pytest `tmp_path`。

验收标准:

- 图片文件存在。
- 图片文件大小大于 0。
- 测试过程不弹出 GUI, 不阻塞。

### 用户阅读文档

场景: 用户本地构建 docs 并浏览安装、快速开始、示例和 API 页面。

验证方式:

```bash
sphinx-build -b html docs docs/_build/html
```

验收标准:

- Sphinx 构建成功。
- `docs/_build/html/index.html` 存在。
- 文档导航包含安装、快速开始、示例和 API 入口。

### 维护者修改代码

场景: 维护者修改代码后运行测试和风格检查。

验证方式:

```bash
pytest
ruff check .
ruff format --check .
```

验收标准:

- pytest 全部通过。
- Ruff check 通过, 或遗留问题被明确记录并限定范围。
- Ruff format check 通过, 或格式化范围被明确记录。

## 输入与输出

### 输入

- 当前源码包 `freeplot/`。
- 新增的 `pyproject.toml`。
- 新增的 `docs/` 文档。
- 新增的 `tests/` 测试。
- README 与文档示例图片资源。

### 输出

- 可执行的测试集合。
- 可执行的安装、导入、Ruff 和 Sphinx 验证命令。
- 验证失败时可定位到安装、导入、绘图、文档或风格检查问题。

## 测试设计

### `tests/test_import.py`

验证内容:

- `import freeplot`。
- `from freeplot import FreePlot`。
- `freeplot.__version__` 存在。

验收标准:

- 导入无异常。
- `FreePlot` 可访问。
- 版本号是非空字符串。

### `tests/test_basic_plot.py`

验证内容:

- 使用 `Agg` 后端。
- 创建 `FreePlot(shape=(1, 1))`。
- 绘制最小 line plot。
- 保存图片到 `tmp_path`。

验收标准:

- 测试无异常。
- 输出图片存在且非空。

### `tests/test_docs_examples.py`

验证内容:

- 覆盖 docs quickstart 中的最小代码路径。
- 若示例需要外部数据或复杂依赖, 应拆成可测试的最小版本。

验收标准:

- 文档最小示例可运行。
- 不依赖交互式窗口。

## 边界条件

- Matplotlib 必须使用 `Agg` 后端, 避免 GUI 阻塞。
- 测试生成的图片必须写入临时目录, 不污染仓库。
- 文档示例不应依赖随机输出的具体数值。
- 若示例包含随机数据, 需要设置固定 seed 或只验证运行成功。
- 若 `SciencePlots` 在环境中不可用, 安装验证必须暴露该问题, 不应静默跳过。
- 若 Sphinx autodoc 导入包时触发绘图后端问题, 应在 docs 配置中处理后端。

## 异常与失败场景

### 安装失败

可能原因:

- `pyproject.toml` 元数据错误。
- 依赖名称错误。
- 构建后端配置错误。

处理方式:

- 优先修复项目配置。
- 不降低已确认的 Python 版本或验证标准。

### 导入失败

可能原因:

- `__init__.py` re-export 错误。
- 依赖缺失。
- style 配置在导入期触发异常。

处理方式:

- 保证导入不执行不必要的绘图逻辑。
- 必要时补充最小导入测试定位问题。

### 绘图测试阻塞

可能原因:

- 使用交互式 Matplotlib 后端。
- 示例调用 `show()`。

处理方式:

- 测试中显式设置 `Agg`。
- 文档示例以 `savefig()` 为主。

### 文档构建失败

可能原因:

- RST 语法错误。
- 图片路径错误。
- autodoc 导入失败。

处理方式:

- 修正 RST 和资源路径。
- API 页面初期聚焦稳定模块。

### Ruff 失败

可能原因:

- 现有代码格式问题过多。
- 导入排序和未使用导入较多。

处理方式:

- 优先修复新代码和配置。
- 对旧代码问题分批处理。
- 若遗留问题无法一次性清空, 必须记录范围和原因。

## 性能、兼容性与约束

- 不为旧 Python 版本做兼容性妥协, 基线采用 Python `>=3.9`。
- 测试应保持轻量, 单次运行目标在数秒级。
- 文档构建不应依赖网络。
- 验证命令应能在本地仓库根目录运行。
- 代码风格应遵循 `format-code-style` skill 和项目现有约定。

## 验收命令

完整验收命令:

```bash
pip install -e ".[dev,docs]"
python -c "import freeplot"
pytest
ruff check .
ruff format --check .
sphinx-build -b html docs docs/_build/html
```

最小快速验收命令:

```bash
python -c "import freeplot"
pytest
```

## 最终验收标准

- `pyproject.toml` 可用于 editable install。
- `python -c "import freeplot"` 成功。
- pytest 最小测试集合通过。
- Ruff 检查和格式化检查可执行, 遗留问题如有必须记录清楚。
- Sphinx 文档可构建成功。
- docs 中包含安装、快速开始、示例和 API 入口。
- docs 示例替代独立 demo 目录作为主要学习入口。
- README 与 docs 的 Python 版本、安装命令、项目定位和示例入口一致。

## 未决问题

- 是否删除现有 `demos/` 和 `freeplot/demos/`, 还是仅从 README/docs 中移除入口。
- 是否删除 `setup.py`, 还是保留为兼容提示文件。
- 是否在 API 文档初期只暴露 `FreePlot`, `UnitPlot`, `FreeAxes` 和 `zoo` 的稳定函数。
