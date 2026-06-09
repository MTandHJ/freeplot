# FreePlot 项目规整与文档建设技术规划

## 方案结论

采用“现代化迁移 + 文档示例替代 demo + 最小验证 + 分阶段代码风格规整”的方案。

本阶段不考虑旧版本兼容性, 优先选择更清晰的现代 Python 项目结构。示例不再以独立 `demos/` 目录作为主要交付物, 而是写入 `docs/` 文档体系中, 使文档成为用户学习和维护验证的核心入口。

## 总体策略

1. 先建立现代项目骨架和工具配置。
2. 再建立 Sphinx 文档体系, 将常用绘图示例沉淀到文档页面。
3. 然后补充最小测试和 Ruff 验证。
4. 最后按 `format-code-style` skill 做保守代码风格规整, 保持行为不变。

## 构建与包管理

### 推荐方案

使用 `pyproject.toml` 作为权威构建配置, 采用 `hatchling` 作为构建后端。

### 理由

- 不需要考虑旧版本兼容性, 可以直接迁移到现代构建方式。
- `freerec` 已使用 `hatchling`, 参考成本低。
- 项目元数据、依赖、可选依赖、Ruff 和 pytest 配置可集中维护。

### 处理方式

- 新增 `pyproject.toml`。
- 将项目元数据迁移到 `[project]`。
- 设置 `requires-python = ">=3.9"`。
- 将运行依赖显式声明为 `matplotlib`, `seaborn`, `numpy`, `pandas`, `SciencePlots`。
- 增加可选依赖:
  - `docs`: Sphinx 文档构建依赖。
  - `dev`: pytest、ruff 等开发依赖。
- `setup.py` 不作为权威配置。是否删除留到实施阶段根据风险确认。

## Python 版本

推荐使用 `>=3.9`。

理由:

- 与 `freerec` 当前基线一致。
- 支持现代工具链和依赖生态。
- 不再为 Python 3.6/3.7 做兼容性妥协。

## 文档体系

### 推荐方案

使用 Sphinx 构建中文主文档, 保留 README 英文与中文版本。

### 目录结构

```text
docs/
  conf.py
  index.rst
  installation.rst
  quickstart.rst
  examples.rst
  api/
    index.rst
  _static/
    img/
  requirements.txt
```

### 页面职责

- `index.rst`: 项目定位、核心特性、导航入口。
- `installation.rst`: 安装方式、依赖说明、开发安装。
- `quickstart.rst`: 最小可运行示例。
- `examples.rst`: 常见绘图示例, 替代独立 demo 目录的学习职责。
- `api/index.rst`: API 参考入口, 初期可使用 autodoc 指向核心模块。

### 示例策略

- 文档中覆盖折线图、散点图、柱状图、热力图、雷达图、小提琴图、局部放大图和 3D 曲面图。
- 示例代码尽量短, 每个示例说明输入、关键 API 和输出效果。
- 示例图片统一放入 `docs/_static/img/`。
- 不再维护独立 demo 目录作为主要学习入口。

## README 策略

README 只保留快速判断和快速上手信息:

- 项目简介。
- 安装方式。
- 3 到 5 行最小示例。
- 文档入口。
- 示例图预览。

README 与 docs 必须保持一致:

- 安装命令一致。
- Python 版本一致。
- 项目定位一致。
- 示例入口一致。

## 测试方案

### 推荐方案

新增 `tests/`, 建立最小回归测试集合。

### 覆盖内容

- `import freeplot` 成功。
- `FreePlot` 可创建基础图。
- 基础绘图方法可保存图片到临时目录。
- 文档中的最小示例可在无 GUI 环境下运行。

### 测试约束

- 使用 Matplotlib `Agg` 后端。
- 不做像素级图像断言。
- 测试优先清晰, 允许少量重复。
- 不为所有绘图函数建立完整矩阵。

## 代码风格规整

### 推荐方案

使用 `format-code-style` skill 作为代码风格依据, 使用 Ruff 执行格式化和基础 lint。

### 执行原则

- 先检查项目现有风格。
- 非平凡代码修改前读取 `references/style.md`。
- 行为保持不变。
- 不为了风格偏好做无关重构。
- 项目已有约定与 skill 冲突时, 默认优先项目约定。

### 规整重点

- 统一导入顺序。
- 补充核心公开模块的 `__all__`。
- 公共 API 逐步补充类型标注。
- 重要类和入口函数使用简洁 NumPy-style docstring。
- 删除明显多余空行和无意义注释。
- 保持核心绘图逻辑不变。

## Ruff 配置

在 `pyproject.toml` 中配置 Ruff:

- `line-length = 101`
- `target-version = "py39"`
- 启用基础规则: `E`, `F`, `W`, `I`
- 暂时忽略 `E501`, 避免一次性处理大量长 docstring。
- 对 `__init__.py` 允许 re-export 相关规则。

格式化与检查命令:

```bash
ruff check .
ruff format --check .
ruff format .
```

若使用 skill 脚本:

```bash
/Users/mtandhj/Desktop/freeskill/skills/format-code-style/scripts/ruff_check.sh .
/Users/mtandhj/Desktop/freeskill/skills/format-code-style/scripts/ruff_format.sh .
```

## demo 目录处理

### 推荐方案

不再将 `demos/` 或 `freeplot/demos/` 作为主要交付物。

实施时有两个可选处理:

- 方案 A: 删除重复 demo 目录, 将有价值的示例迁移到 docs。
- 方案 B: 暂时保留 demo 目录, 但 README 和 docs 不再指向它们。

推荐方案 A, 但实施前需要确认是否允许删除现有 demo 文件。

## 方案取舍

| 事项 | 方案 A | 方案 B | 推荐 |
|---|---|---|---|
| 构建后端 | `hatchling` | `setuptools` 过渡 | `hatchling` |
| Python 版本 | `>=3.9` | `>=3.8` | `>=3.9` |
| 文档语言 | 中文主文档 | 中英双语一次到位 | 中文主文档 |
| 示例载体 | `docs/examples.rst` | 独立 `demos/` | `docs/examples.rst` |
| `setup.py` | 删除或弱化 | 继续作为权威 | 删除或弱化 |
| CI | 暂不引入 | 立即引入 GitHub Actions | 暂不引入 |

## 验证命令

```bash
pip install -e ".[dev,docs]"
python -c "import freeplot"
pytest
ruff check .
ruff format --check .
sphinx-build -b html docs docs/_build/html
```

## 风险与应对

- `SciencePlots` 依赖名称和导入行为可能影响安装验证。应在实施时用 editable install 验证。
- Matplotlib 后端可能导致测试阻塞。测试入口应显式使用 `Agg`。
- 删除 demo 目录可能影响用户已有引用。若不能确认删除, 先保留但从 README/docs 中移除入口。
- API 文档自动生成可能暴露当前 docstring 风格问题。初期 API 页面可以聚焦核心模块, 后续逐步补齐。
- Ruff 一次性修复可能产生大量 diff。应先运行 check, 再分批格式化核心包和测试。

## 验收标准

- `pyproject.toml` 成为项目权威配置。
- `pip install -e ".[dev,docs]"` 可成功执行。
- `python -c "import freeplot"` 可成功执行。
- `pytest` 可通过最小测试集合。
- `ruff check .` 可通过, 或遗留问题被明确记录。
- `ruff format --check .` 可通过, 或格式化范围被明确记录。
- `sphinx-build -b html docs docs/_build/html` 可成功构建。
- README 与 docs 的安装方式、Python 版本、项目定位和示例入口一致。
- 常见绘图示例已进入 docs, 不再依赖独立 demo 目录作为主要入口。
