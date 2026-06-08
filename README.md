# 神经网络的谱偏差及其在 Helmholtz 方程求解中的影响

本压缩包包含一篇可直接提交的中文课程论文及其可复现实验材料。

## 文件结构

- `main.pdf`：已编译好的论文 PDF。
- `main.tex`：论文 LaTeX 源码，建议使用 XeLaTeX 编译。
- `references.bib`：BibTeX 参考文献数据库，包含正文引用的 20 篇文献。
- `code/run_experiments.py`：生成数值实验结果、表格与图像的 Python 代码。
- `figures/`：论文中使用的实验图像，含 PDF 与 PNG 版本。
- `data/`：实验生成的数据与结果表。
- `requirements.txt`：运行实验代码所需的 Python 依赖。

## 编译论文

由于参考文献已改为 `.bib` 文件管理，推荐按如下顺序编译：

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

如果使用 TeX 编辑器，请选择 `XeLaTeX + BibTeX` 工作流。

## 重新运行实验

```bash
pip install -r requirements.txt
python code/run_experiments.py
```

运行后会更新 `figures/` 与 `data/` 中的图表和数据。

## 提交前需要修改

论文首页保留了“姓名、学号”的横线，请提交前填入自己的信息。

本版本已将参考文献整理为独立的 `references.bib` 文件，并同步修改了 `main.tex` 中的参考文献调用方式。
