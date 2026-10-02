# 多范数等腰三角形逼近

[English README](README.md)

本仓库包含等腰三角形逼近问题的数值代码。对于具有三条正边长的输入三角形，
程序在给定角度/系数约束下考察可行的等腰目标三角形，并比较 $\ell_1$、
$\ell_2$ 和 $\ell_\infty$ 范数下的最优残差。代码可生成决策区域与代价图，
并在周长为 1 的三角形网格上穷尽验证公式。

这些程序实现配套论文采用的数值模型，不是通用几何软件库。

## 仓库内容

### 源代码文件

| 文件 | 用途 |
| --- | --- |
| `generate_figures.py` | 生成论文风格的决策区域图和代价图。 |
| `verify_multinorm_grid.py` | 唯一受支持的穷尽数值验证程序。 |
| `requirements.txt` | 运行依赖。 |
| `README.md`、`README_zh.md` | 英文和简体中文说明。 |

`verify_multinorm.py` 是旧的、重复的验证脚本。请**不要使用，也不要上传**到公开
GitHub 仓库。

### 生成的输出

`generate_figures.py` 会生成成对的 PDF 和 PNG 文件。使用下方推荐命令时，
$\ell_1$ 输出为：

```text
1-90_decision_regions.{pdf,png}       1-90_costs.{pdf,png}
45-90_decision_regions.{pdf,png}      45-90_costs.{pdf,png}
30-50_decision_regions.{pdf,png}      30-50_costs.{pdf,png}
90-120_decision_regions.{pdf,png}     90-120_costs.{pdf,png}
balance_decision_regions.{pdf,png}    balance_costs.{pdf,png}
```

使用 `--paper --english` 时，还会生成：

```text
1-90_l2_decision_regions.{pdf,png}    1-90_l2_costs.{pdf,png}
1-90_linf_decision_regions.{pdf,png}  1-90_linf_costs.{pdf,png}
```

除非传入 `--no-results-json`，绘图程序还会写入
`figure_generation_results.json`。验证程序会写入
`verification_results.json` 和 `verification_summary.tex`。

生成的图片、JSON 记录和 TeX 摘要均为输出文件，不是源代码；独立的代码仓库不需要
包含它们。可选择保留 `verification_results.json`，作为可复现性记录。

## 前提条件与安装

请使用 Python 3.8 至 3.11。SciPy 的上限是有意设置的：独立线性规划控制使用
SciPy 的 `revised simplex` 方法，该方法已在更新的 SciPy 版本中移除。

```bash
conda activate base
python -m pip install -r requirements.txt
```

所需包为 NumPy、SciPy 和 Matplotlib。除环境管理外，不强制要求使用 Conda；
任何兼容的 Python 环境均可运行。

## 复现图形

请在脚本所在目录运行：

```bash
python generate_figures.py --paper --balance --english --output-dir .
```

该命令会在当前目录生成上列十个 $\ell_1$ 文件和四个
$\ell_2$/$\ell_\infty$ 文件，以及 `figure_generation_results.json`。
$\ell_1$ 图对的默认输出目录为 `pic/`。

### 绘图命令行参数

| 参数 | 含义 |
| --- | --- |
| `--paper` | 选择四个标准角度区间：`[1°,90°]`、`[45°,90°]`、`[30°,50°]` 和 `[90°,120°]`；这是默认区间集合。 |
| `--balance` | 额外生成平衡系数情形 `(k1,k2)=(0.75,1.25)`。 |
| `--alpha A --beta B` | 生成一个明确指定的角度区间 `[A,B]`（单位为度）；两个参数必须同时提供，且不能与 `--paper` 合用。 |
| `--interactive` | 以交互方式输入 `alpha` 和 `beta`（单位为度）。 |
| `--step S` | 输入三角形格点间距，不是目标边长离散步长。默认值：`0.01`。 |
| `--surf-n N` | 插值曲面网格每边的划分数。默认值：`220`。 |
| `--output-dir DIR` | $\ell_1$ 图对和 `figure_generation_results.json` 的输出目录。默认值：`pic/`。 |
| `--english` / `--russian` | 使用英文或俄文图题、图例和色标；均不指定时使用中文。两个参数不能同时使用。 |
| `--hide-tie-lines` | 不绘制决策图中的并列线段，也不在图例中列出。 |
| `--output-stem STEM` | 替换所有 $\ell_1$ 输出图对的文件名前缀。 |
| `--no-results-json` | 不写入 `figure_generation_results.json`。 |

$\ell_2$ 和 $\ell_\infty$ 图仅在使用 `--paper --english` 时生成，且会写入
`generate_figures.py` 所在目录；这是脚本当前的既定行为。

## 穷尽验证

请只运行受支持的验证程序：

```bash
python verify_multinorm_grid.py
```

验证使用周长 1、步长 `0.01` 的输入网格，保留严格三角形不等式和所有有序、
保持位置的边长三元组：**五组设置中的每一组均有 1,176 个三元组**：

1. `[1°,90°]`
2. `[45°,90°]`
3. `[30°,50°]`
4. `[90°,120°]`
5. `(k1,k2)=(3/4,5/4)`

对每个三元组，程序检查三个底边分支以及双分支全局规则。独立验证控制如下：

| 范数 | 公式验证控制 |
| --- | --- |
| $\ell_1$ | 有限断点枚举和独立线性规划。 |
| $\ell_2$ | 对降维平方目标进行直接连续最小化。 |
| $\ell_\infty$ | 独立的上图（epigraph）线性规划。 |

验证程序将公式/控制误差、可行性违背、重构误差和 $\ell_1$ 并列计数写入
`verification_results.json`，并生成 TeX 宏文件 `verification_summary.tex`。

## 建议公开上传到 GitHub 的文件

只上传：

```text
generate_figures.py
verify_multinorm_grid.py
requirements.txt
README.md
README_zh.md
```

可选择上传 `verification_results.json` 作为已完成运行的记录。不要上传
`verify_multinorm.py`、生成的图形、论文 LaTeX 源码、出版社模板/类文件、参考文献
文件、PDF 或其他论文材料；独立的数值代码仓库不需要这些文件。
