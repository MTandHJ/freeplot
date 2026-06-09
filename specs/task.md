# FreePlot 项目规整与文档建设任务拆解

## 状态说明

- `[ ]`: 未开始
- `[~]`: 进行中
- `[x]`: 已完成
- `[!]`: 阻塞或需要确认

## 任务总览

### 1. 项目基线检查

- [x] 检查当前 git 状态, 识别已有改动, 避免覆盖用户修改。
- [x] 读取核心文件: `setup.py`, `freeplot/__init__.py`, `freeplot/base.py`, `freeplot/unit.py`, `freeplot/utils.py`, `freeplot/config.py`, `freeplot/zoo.py`。
- [x] 检查现有 README、图片资源和 demo 文件, 识别可迁移到 docs 的示例内容。
- [x] 确认 `format-code-style` skill 的 `references/style.md` 和 `references/ruff.md` 已用于指导代码风格。

### 2. 构建配置现代化

- [x] 新增 `pyproject.toml`, 使用 `hatchling` 作为构建后端。
- [x] 迁移项目元数据: name、version、description、readme、license、authors、Python 版本和 classifiers。
- [x] 声明运行依赖: `matplotlib`, `seaborn`, `numpy`, `pandas`, `SciencePlots`。
- [x] 声明可选依赖: `dev` 和 `docs`。
- [x] 添加 Ruff 配置。
- [x] 添加 pytest 配置。
- [!] 决定并处理 `setup.py`: 暂时保留, 不再作为权威配置入口。

### 3. 文档体系建设

- [x] 新建 `docs/` 目录结构。
- [x] 新建 `docs/conf.py`。
- [x] 新建 `docs/requirements.txt`。
- [x] 新建 `docs/index.rst`, 写入项目定位、核心特性和导航。
- [x] 新建 `docs/installation.rst`, 写入安装和开发安装说明。
- [x] 新建 `docs/quickstart.rst`, 写入最小可运行示例。
- [x] 新建 `docs/examples.rst`, 写入常见绘图示例。
- [x] 新建 `docs/api/index.rst`, 提供 API 参考入口。
- [x] 整理 `docs/_static/img/` 图片资源。

### 4. 文档示例迁移

- [x] 从现有 README 和 demo 中提取折线图示例。
- [x] 提取散点图示例。
- [x] 提取柱状图示例。
- [x] 提取热力图示例。
- [x] 提取雷达图示例。
- [x] 提取小提琴图示例。
- [x] 提取局部放大图示例。
- [x] 提取 3D 曲面图示例。
- [x] 将示例统一改为文档代码片段, 避免依赖交互式 `show()`。
- [x] 确保示例输出图路径与 docs 资源路径一致。

### 5. README 整理

- [x] 更新 `README.md`, 保留英文快速介绍、安装、最小示例、文档入口和示例预览。
- [x] 更新 `README_zh_CN.md`, 与英文 README 和 docs 保持信息一致。
- [x] 移除或弱化 README 中对独立 `demos/` 的入口依赖。
- [x] 校验 README 图片路径仍有效。

### 6. demo 目录处理

- [!] 确认是否删除现有 `demos/` 和 `freeplot/demos/`。
- [ ] 若允许删除, 删除重复 demo 目录。
- [x] 若暂不删除, 保留文件但从 README/docs 中移除主要入口。
- [x] 确保包构建不把 `freeplot/demos/` 作为核心 API 文档入口。

### 7. 最小测试建设

- [x] 新建 `tests/` 目录。
- [x] 新建 `tests/test_import.py`, 覆盖包导入、`FreePlot` 导出和版本号。
- [x] 新建 `tests/test_basic_plot.py`, 使用 `Agg` 后端创建并保存基础图。
- [x] 新建 `tests/test_docs_examples.py`, 覆盖 docs quickstart 的最小示例路径。
- [x] 确保测试输出写入 `tmp_path`, 不污染仓库。

### 8. 代码风格规整

- [x] 运行 Ruff check, 获取初始问题列表。
- [x] 按 `format-code-style` 原则规整新文件。
- [x] 对核心公开模块做保守规整: 导入顺序、`__all__`, 明显空行和格式问题。
- [x] 保持核心绘图逻辑不变。
- [x] 若旧代码 Ruff 问题过多, 记录遗留范围, 避免为清空 lint 做大规模重构。

### 9. 本地验证

- [!] 执行 `pip install -e ".[dev,docs]"`: base 环境缺少 `hatchling`, 且联网安装未授权。
- [x] 执行 `python -c "import freeplot"`。
- [x] 执行 `pytest`。
- [x] 执行 `ruff check .`。
- [x] 执行 `ruff format --check .`。
- [!] 执行 `sphinx-build -b html docs docs/_build/html`: base 环境缺少 `sphinx`, 且联网安装未授权。
- [x] 记录无法执行或失败的命令、原因和后续处理建议。

### 10. 收尾检查

- [x] 检查 `specs/spec.md`, `specs/plan.md`, `specs/validation.md`, `specs/task.md` 前后一致。
- [x] 检查 README 与 docs 的安装命令、Python 版本、项目定位和示例入口一致。
- [x] 检查 git diff, 确认没有无关改动。
- [x] 汇总完成项、验证结果、遗留问题和建议下一步。

## 验证结果

- [x] `MPLCONFIGDIR=/private/tmp/freeplot-mpl conda run -n base python -c "import freeplot; print(freeplot.__version__)"`
- [x] `MPLCONFIGDIR=/private/tmp/freeplot-mpl conda run -n base python -m pytest`
- [x] `conda run -n base python -m ruff check .`
- [x] `conda run -n base python -m ruff format --check .`
- [!] `python -m pip install -e ".[dev,docs]"`: 需要下载 `hatchling`, 当前联网安装未授权。
- [!] `sphinx-build -b html docs docs/_build/html`: base 环境缺少 `sphinx`。

## 实施顺序

推荐按以下顺序实施:

1. 项目基线检查。
2. 构建配置现代化。
3. 文档体系建设。
4. 文档示例迁移。
5. README 整理。
6. 最小测试建设。
7. 代码风格规整。
8. 本地验证。
9. demo 目录处理确认和收尾。

## 需要确认的问题

- 是否允许删除 `demos/` 和 `freeplot/demos/`。
- 是否删除 `setup.py`, 还是保留为兼容提示文件。
- API 文档初期是否只暴露 `FreePlot`, `UnitPlot`, `FreeAxes` 和 `zoo` 的稳定函数。
