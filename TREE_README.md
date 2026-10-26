# Decision tree from scratch

A Python classification tree implemented with the standard library only. No
scikit-learn, NumPy, pandas, or existing tree algorithm is used or required.
Python 3.10 or newer is required. This assignment is separate from the older
competition scripts `train.py` and `optimize.py`.

## Quick start

Run from the `assignment_1` directory, without installing dependencies:

```python
from entropy_tree import DecisionTreeClassifier

X = [[22, "sunny"], [18, "rain"], [30, "sunny"], [16, "rain"]]
y = ["play", "stay", "play", "stay"]
tree = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=5)
tree.fit(X, y, feature_names=["temperature", "weather"])
print(tree.predict([[25, "sunny"], [None, "rain"]]))
print(tree.predict_proba([[25, "sunny"]]))  # columns follow tree.classes_
print(tree.export_text())
```

## Any tabular classification dataset

Pass a rectangular sequence of rows and a matching sequence of class labels.
Features may be numbers, strings, booleans, or missing values (`None` / numeric
NaN). Labels may be nonmissing strings, booleans, or finite numbers. Multiple
classes are supported. Each row must use the same feature order.

Numeric columns are inferred when every observed value is a non-boolean number;
other columns are categorical. Override inference with, for example,
`fit(X, y, feature_types=["numeric", "categorical"])` for numeric category codes.
All-missing columns default to numeric and cannot produce a split on their own.
NumPy arrays and pandas DataFrames are accepted if supplied by the caller, but
neither library is imported or needed. For DataFrames, prediction requires the
same column names and order as fitting.

For CSV files, specify your own target column:

```powershell
python -m entropy_tree examples/weather.csv --target play --max-depth 3 --max-leaf-nodes 5 --show-tree
python -m entropy_tree your_dataset.csv --target label --drop id --categorical postcode --max-depth 8
```

The command reserves a reproducible random 20% holdout (`--seed 42`, adjustable
with `--test-size`). Types are inferred from training rows only. Empty feature
cells become missing values. CSV labels remain strings. If a numeric-looking
training column contains text in the holdout, use `--categorical` for that column.
The split is not stratified; small or imbalanced datasets may need a custom split.
The CLI reports accuracy, depth, leaf count, and optionally the tree.

This is a general **tabular classifier**, not a regression, text, image, or
time-series pipeline. Convert raw data to suitable features first. Infinite
numbers, ragged rows, missing labels, and nested feature objects are rejected.

## Tree size

| Parameter | Default | Meaning |
|---|---|---|
| `max_depth` | `10` | Maximum edges from root to leaf; `0` gives one leaf, `None` removes the limit |
| `max_leaf_nodes` | `None` | Maximum leaves; `L` leaves imply at most `2L - 1` total nodes |
| `min_samples_split` | `2` | Minimum samples for a node to attempt a split |
| `min_samples_leaf` | `1` | Minimum samples in each child of a split |
| `min_gain` | `0.0` | Minimum local information gain in bits; zero-gain splits are always rejected |

`depth_`, `n_leaves_`, and `node_count_` describe the fitted size. When a leaf budget
is supplied, the next split maximizes the sample-weighted entropy reduction
across available leaves. Ties are deterministic. No recursion is used to build,
predict, or display the tree.

## Algorithm and missing values

Entropy is `H(S) = -sum(p(c) * log2(p(c)))` over classes. For each candidate split,
information gain is `H(parent) - n_left/n * H(left) - n_right/n * H(right)`.
The algorithm selects the largest positive gain (above numerical tolerance).

Numeric candidates separate every pair of distinct sorted observed values, using
the lower observed value as the `<=` threshold. This avoids overflow and defines
how unseen values between observations are routed. Categorical candidates test
one category against the rest. Missing values are tried on both branches, and
the better route is retained. A missing-versus-observed candidate is also tried.
Without training missing values, future missing values follow the larger child
(left on ties). Unseen categories follow the unequal branch.

Leaves predict their majority class; ties use first class appearance in the
training labels. Probabilities are the leaf's class proportions, not calibrated
confidence estimates. Growth stops at pure nodes, configured limits, or when no
valid positive-gain split exists. In particular, a pure XOR interaction cannot
be learned without another informative feature or allowing zero-gain splits.

## Tests and packaging

```powershell
python -m unittest discover -s tests -v
python -m pip install .
entropy-tree examples/weather.csv --target play --show-tree
```

The package has no runtime dependencies. Packaging uses setuptools. To build
distribution files and optionally publish under your own PyPI account:

```powershell
python -m pip install build twine
python -m build
python -m twine check dist/*
# Only when ready to publish, with your own credentials:
python -m twine upload dist/*
```

The distribution name is `epita-entropy-tree`; its PyPI availability has not been
verified. Change it in `pyproject.toml` if necessary. Packaging is provided;
publication to PyPI has not been performed.
