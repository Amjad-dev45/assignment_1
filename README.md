# Assignment 1: Decision tree from scratch

A pure Python classifier using entropy and information gain, with numeric and
categorical features, missing-value handling, tree-size controls, and text display.
No existing decision tree library is used. Python 3.10+ is required.

See [the report](report.md) and [full documentation](TREE_README.md).

```sh
python -m entropy_tree examples/weather.csv --target play --max-depth 3 --show-tree
python -m unittest discover -s tests -v
```

Run these commands from the repository root. The project is packaged for local
installation with `python -m pip install .`; it has not been published to PyPI.
