# 高维统计推断：稀疏回归变量选择方法对比研究

A comprehensive comparative study of LASSO, Elastic Net, and SCAD for variable selection in high-dimensional sparse linear regression.

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)

## 核心发现 / Key Findings

| 方法 / Method | 平均MSE | TPR | FDR |
|---------|---------|-----|-----|
| LASSO | 78.8 | 0.950 | 0.111 |
| **Elastic Net** | **71.2** | **0.973** | **0.069** |
| SCAD | 76.0 | 0.953 | 0.090 |

**Elastic Net shows best overall performance** ⭐

## 快速开始 / Quick Start

```bash
git clone https://github.com/qq2684/high-dimensional-regression
cd high-dimensional-regression
pip install -r requirements.txt
python run_project.py
```

## 项目文件 / Files

- **code/** - Core implementations (data generation, methods, experiments)
- **results/** - Experimental results (18 configurations)
- **ACADEMIC_REPORT.md** - Full academic paper (~8,000 words)
- **README_EN.md** - Complete English documentation
- **GITHUB_GUIDE.md** - GitHub publishing guide

## 许可证 / License

MIT License - See [LICENSE](LICENSE)

**More information**: See [ACADEMIC_REPORT.md](ACADEMIC_REPORT.md) for full details.
