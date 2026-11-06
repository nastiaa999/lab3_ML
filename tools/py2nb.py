"""Собирает ноутбук из .py с ячейками «# %%» и выполняет его.

Запуск: python tools/py2nb.py 03_analysis.py  ->  03_analysis.ipynb (с результатами всех ячеек)
"""
import re
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

src_path = sys.argv[1]
text = open(src_path, encoding="utf-8").read()
cells = []
for chunk in re.split(r"^# %%", text, flags=re.M)[1:]:
    if chunk.startswith(" [markdown]"):
        body = chunk[len(" [markdown]"):].strip("\n")
        cells.append(nbformat.v4.new_markdown_cell("\n".join(line[2:] for line in body.splitlines())))
    else:
        cells.append(nbformat.v4.new_code_cell(chunk.strip("\n")))
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
ExecutePreprocessor(timeout=600).preprocess(nb, {"metadata": {"path": "."}})
nbformat.write(nb, src_path.replace(".py", ".ipynb"))
print("готово:", src_path.replace(".py", ".ipynb"))
