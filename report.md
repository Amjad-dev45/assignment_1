# Decision tree implementation report

## Objective

Implement a Python decision tree classifier from scratch using entropy and
information gain, with a tree-size hyperparameter and a way to display the tree.
GitHub repository: [Amjad-dev45/assignment_1](https://github.com/Amjad-dev45/assignment_1).

## Work completed

The new `entropy_tree` package contains the complete training and prediction
algorithm, implemented using only Python's standard library. No existing tree
implementation is called. The older competition scripts in the parent folder are
unrelated to this submission and are not imported by the package.

- `entropy_tree/tree.py`: entropy, split search, best-first tree growth, input
  validation, predictions, class probabilities, accuracy, and text display.
- `entropy_tree/__main__.py`: CSV loading, configurable target and feature columns,
  reproducible holdout evaluation, and optional display.
- `tests/test_tree.py`: standard-library unit tests, including an independent
  brute-force check of numeric split information gains.
- `examples/weather.csv`: a small mixed-feature example dataset.
- `TREE_README.md`: API, CLI examples, design decisions, limitations, and packaging.
- `pyproject.toml`: installable package with an `entropy-tree` command and no
  runtime dependencies.

## Method

For class proportions p(c), entropy is:

`H(S) = -sum_c p(c) log2(p(c))`.

For each binary split into L and R, information gain is:

`IG = H(S) - (|L| / |S|) H(L) - (|R| / |S|) H(R)`.

Numeric values are sorted, and each distinct boundary is evaluated using running
class counts. The threshold is the lower observed boundary value. Categorical
features use category-equality splits. Missing-value routing is learned by
evaluating both branches; missing-versus-present splits are also supported.
Each node chooses its highest positive-gain valid split. A priority queue expands
the available leaf with the greatest sample-weighted reduction in entropy.

Leaves store class counts and predict the majority class. Class ties use first
appearance in the training labels. Unseen categories take the unequal branch;
unseen missing values follow the learned route or the larger training child.

## Hyperparameters and bonus features

`max_depth` directly controls tree depth (root depth is zero). `max_leaf_nodes`
provides an additional global size bound: at most L leaves and 2L - 1 total nodes.
`min_samples_split`, `min_samples_leaf`, and `min_gain` restrict growth further.

The display bonus is implemented with `export_text()`, showing feature conditions,
branches, missing-value routes, sample counts, entropy, gain, and leaf predictions.

The PyPI bonus is partially completed: the package is installable and distribution
metadata and publishing commands are provided. It has **not been published** to
PyPI, and no public package URL is claimed.

## Validation

Verified locally with Python 3.12: all 16 unit tests passed. The CSV example
completed with 11 training rows and 3 holdout rows, accuracy 0.3333, depth 3,
4 leaves, and 7 total nodes. This tiny example verifies the workflow and is not
a performance benchmark. A wheel was built successfully and installed into an
isolated local directory; a prediction smoke test against that installation passed.

Run `python -m unittest discover -s tests -v`. Tests cover known entropy values,
perfect splits, numeric split gains against brute-force enumeration, multiclass
categorical data, missing and constant features, unseen categories, class ties,
probabilities, extreme numeric values, size limits, minimum sample counts,
invalid inputs, refitting, deterministic behavior, display, and zero-gain stopping.

Run the end-to-end CSV example:

```powershell
python -m entropy_tree examples/weather.csv --target play --max-depth 3 --max-leaf-nodes 5 --show-tree
```

## Scope and limitations

The code accepts general rectangular tabular classification datasets, including
mixed numeric/categorical features and any number of classes. It is not a
regressor or a raw image/text processor. Data must fit in memory. The CSV holdout
is random rather than stratified, so accuracy on small samples is illustrative.

For a node with n samples, d numeric features, and k classes, numeric split search
costs approximately O(d * (n log n + n k)); categorical work also depends on the
number of distinct categories. Sorting repeats at each node. The implementation
prioritizes clarity over the performance of compiled libraries; deep trees or
large datasets can be slow. Category-versus-rest splitting does not exhaustively
search all category subsets. Positive-gain-only greedy growth cannot discover
pure XOR. Depth and leaf limits help control overfitting, but should be selected
using validation data rather than training accuracy.
