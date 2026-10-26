"""Binary classification trees, implemented using only the Python standard library."""

from collections import Counter
from dataclasses import dataclass
import heapq
import math
from numbers import Real


def _entropy_counts(counts):
    total = sum(counts)
    return -sum((n / total) * math.log2(n / total) for n in counts if n) if total else 0.0


def entropy(labels):
    """Shannon entropy in bits; the empty sequence has entropy zero."""
    return _entropy_counts(Counter(labels).values())


def _missing(value):
    return value is None or (isinstance(value, Real) and math.isnan(value))


def _matrix(X):
    # Optional compatibility with DataFrames and NumPy arrays, without importing them.
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    if hasattr(X, "tolist"):
        X = X.tolist()
    if isinstance(X, (str, bytes, dict)):
        raise ValueError("X must be a two-dimensional sequence of rows")
    try:
        rows = []
        for row in X:
            if isinstance(row, (str, bytes, dict)):
                raise ValueError("Each row must be a sequence of feature values")
            rows.append(list(row))
        return rows
    except TypeError as exc:
        raise ValueError("X must be a two-dimensional sequence of rows") from exc


@dataclass
class Node:
    """A fitted node; children are None for a leaf."""

    prediction: object
    counts: tuple
    n_samples: int
    impurity: float
    depth: int
    feature: int | None = None
    kind: str | None = None
    value: object = None
    missing_left: bool = True
    gain: float = 0.0
    left: object = None
    right: object = None

    @property
    def is_leaf(self):
        return self.left is None


class DecisionTreeClassifier:
    """Greedy entropy tree for tabular classification.

    Numeric features use <= thresholds. Categorical features use equality versus
    all other values. None and NaN are missing; their branch is learned at fit time.
    Root depth is zero. max_leaf_nodes bounds leaves (and total nodes to 2L - 1).
    Numeric category codes can be declared categorical with feature_types at fit.
    """

    def __init__(self, max_depth=10, max_leaf_nodes=None, min_samples_split=2,
                 min_samples_leaf=1, min_gain=0.0):
        for name, value, minimum, optional in [
            ("max_depth", max_depth, 0, True),
            ("max_leaf_nodes", max_leaf_nodes, 1, True),
            ("min_samples_split", min_samples_split, 2, False),
            ("min_samples_leaf", min_samples_leaf, 1, False),
        ]:
            if value is None and optional:
                continue
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}" +
                                 (" or None" if optional else ""))
        if not isinstance(min_gain, Real) or not math.isfinite(min_gain) or min_gain < 0:
            raise ValueError("min_gain must be finite and nonnegative")
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain

    def _validate_values(self, rows, types):
        for row in rows:
            if len(row) != len(types):
                raise ValueError("All rows must have the same number of features as training")
            for value, kind in zip(row, types):
                if _missing(value):
                    continue
                if not isinstance(value, (str, Real)):
                    raise ValueError("Features must be strings, real numbers, booleans, None or NaN")
                if isinstance(value, Real) and not math.isfinite(value):
                    raise ValueError("Infinite feature values are not supported")
                if kind == "numeric" and not isinstance(value, Real):
                    raise ValueError("Numeric columns must contain numbers or missing values")

    def fit(self, X, y, feature_names=None, feature_types=None):
        """Fit rows X to discrete labels y; return self.

        feature_types optionally specifies 'numeric' or 'categorical' per column.
        Tied labels follow first appearance in y; split ties follow column order.
        """
        inferred_names = list(X.columns) if hasattr(X, "columns") else None
        rows = _matrix(X)
        labels = list(y)
        if not rows or not rows[0] or len(rows) != len(labels):
            raise ValueError("X and y must be nonempty, with equal sample counts and at least one feature")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("X must be rectangular")
        for label in labels:
            if not isinstance(label, (str, Real)) or _missing(label):
                raise ValueError("Labels must be nonmissing strings, booleans or finite numbers")
            if isinstance(label, Real) and not math.isfinite(label):
                raise ValueError("Labels must be finite")
        if feature_types is None:
            types = ["numeric" if all(_missing(row[j]) or
                     (isinstance(row[j], Real) and not isinstance(row[j], bool))
                     for row in rows) else "categorical" for j in range(width)]
        else:
            types = list(feature_types)
        if len(types) != width or any(t not in ("numeric", "categorical") for t in types):
            raise ValueError("feature_types must contain one 'numeric' or 'categorical' per column")
        names = list(feature_names) if feature_names is not None else inferred_names
        if names is None:
            names = [f"x[{j}]" for j in range(width)]
        if len(names) != width:
            raise ValueError("feature_names must match the number of columns")
        self._validate_values(rows, types)
        self.classes_ = list(dict.fromkeys(labels))
        codes = {label: i for i, label in enumerate(self.classes_)}
        targets = [codes[label] for label in labels]
        self.n_features_in_ = width
        self.feature_names_ = [str(name) for name in names]
        self.feature_types_ = types

        def make_node(indices, depth):
            counts = Counter(targets[i] for i in indices)
            frequencies = tuple(counts[c] for c in range(len(self.classes_)))
            prediction = self.classes_[max(range(len(frequencies)), key=frequencies.__getitem__)]
            return Node(prediction, frequencies, len(indices), _entropy_counts(frequencies), depth)

        self.root_ = make_node(list(range(len(rows))), 0)
        self.n_leaves_ = 1
        self.node_count_ = 1
        self.depth_ = 0
        queue = []
        serial = 0

        def enqueue(node, indices):
            nonlocal serial
            if (node.impurity == 0 or len(indices) < self.min_samples_split or
                    (self.max_depth is not None and node.depth >= self.max_depth)):
                return
            split = self._best_split(rows, targets, indices, node)
            if split is not None:
                # Best-first growth: maximize reduction in whole-training-set entropy.
                heapq.heappush(queue, (-split[0] * len(indices), serial, node, indices, split))
                serial += 1

        if self.max_leaf_nodes != 1:
            enqueue(self.root_, list(range(len(rows))))
        while queue and (self.max_leaf_nodes is None or self.n_leaves_ < self.max_leaf_nodes):
            _, _, node, indices, split = heapq.heappop(queue)
            node.gain, node.feature, node.kind, node.value, node.missing_left = split
            left, right = [], []
            for i in indices:
                (left if self._go_left(rows[i], node) else right).append(i)
            node.left = make_node(left, node.depth + 1)
            node.right = make_node(right, node.depth + 1)
            self.n_leaves_ += 1
            self.node_count_ += 2
            self.depth_ = max(self.depth_, node.depth + 1)
            if self.max_leaf_nodes is None or self.n_leaves_ < self.max_leaf_nodes:
                enqueue(node.left, left)
                enqueue(node.right, right)
        return self

    def _best_split(self, rows, targets, indices, node):
        total = Counter(targets[i] for i in indices)
        best = None
        best_gain = -math.inf

        def consider(feature, kind, value, left_known, missing):
            nonlocal best, best_gain
            # If no training values are missing, send future missing values to
            # the larger child. Otherwise evaluate both routes explicitly.
            n_missing = sum(missing.values())
            n_left = sum(left_known.values())
            routes = (True, False) if n_missing else (n_left >= len(indices) - n_left,)
            for missing_left in routes:
                left = left_known + missing if missing_left else left_known
                nl = sum(left.values())
                nr = len(indices) - nl
                if min(nl, nr) < self.min_samples_leaf:
                    continue
                right = total - left
                gain = node.impurity - (nl * _entropy_counts(left.values()) +
                        nr * _entropy_counts(right.values())) / len(indices)
                if gain > 1e-12 and gain >= self.min_gain and gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (gain, feature, kind, value, missing_left)

        for j, kind in enumerate(self.feature_types_):
            missing = Counter(targets[i] for i in indices if _missing(rows[i][j]))
            known = [i for i in indices if not _missing(rows[i][j])]
            if missing and known:
                consider(j, "missing", None, Counter(), missing)
            if kind == "numeric":
                ordered = sorted(known, key=lambda i: rows[i][j])
                left = Counter()
                for position in range(len(ordered) - 1):
                    i, next_i = ordered[position], ordered[position + 1]
                    left[targets[i]] += 1
                    a, b = rows[i][j], rows[next_i][j]
                    if a != b:
                        # Lower observed endpoint is an exact, overflow-free
                        # representative of the partition between distinct values.
                        consider(j, "numeric", a, left, missing)
            else:
                groups = {}
                for i in known:
                    groups.setdefault(rows[i][j], Counter())[targets[i]] += 1
                for value, counts in groups.items():
                    consider(j, "categorical", value, counts, missing)
        return best

    @staticmethod
    def _go_left(row, node):
        value = row[node.feature]
        if _missing(value):
            return node.missing_left
        if node.kind == "missing":
            return False
        if node.kind == "numeric":
            return value <= node.value
        return value == node.value

    def _leaves(self, X):
        if not hasattr(self, "root_"):
            raise RuntimeError("Call fit before predicting or displaying the tree")
        if hasattr(X, "columns") and [str(c) for c in X.columns] != self.feature_names_:
            raise ValueError("DataFrame columns must match training feature names and order")
        rows = _matrix(X)
        self._validate_values(rows, self.feature_types_)
        leaves = []
        for row in rows:
            node = self.root_
            while not node.is_leaf:
                node = node.left if self._go_left(row, node) else node.right
            leaves.append(node)
        return leaves

    def predict(self, X):
        """Return a list of class predictions."""
        return [node.prediction for node in self._leaves(X)]

    def predict_proba(self, X):
        """Leaf class frequencies, with columns in classes_ order."""
        return [[n / node.n_samples for n in node.counts] for node in self._leaves(X)]

    def score(self, X, y):
        """Return accuracy on nonempty labeled rows."""
        predictions, labels = self.predict(X), list(y)
        if not labels or len(labels) != len(predictions):
            raise ValueError("y must be nonempty and match the prediction count")
        return sum(a == b for a, b in zip(predictions, labels)) / len(labels)

    def export_text(self):
        """Return a readable tree; print(model.export_text()) displays it."""
        if not hasattr(self, "root_"):
            raise RuntimeError("Call fit before displaying the tree")
        lines = []
        stack = [(self.root_, "", "root: ")]
        while stack:
            node, indent, branch = stack.pop()
            stats = f"n={node.n_samples}, entropy={node.impurity:.4f}"
            if node.is_leaf:
                text = f"predict {node.prediction!r} ({stats}, counts={list(node.counts)})"
            else:
                name = self.feature_names_[node.feature]
                if node.kind == "missing":
                    rule = f"{name} is missing"
                else:
                    operator = "<=" if node.kind == "numeric" else "=="
                    rule = f"{name} {operator} {node.value!r}"
                text = f"{rule} [missing -> {'yes' if node.missing_left else 'no'}] ({stats}, gain={node.gain:.4f})"
                stack.append((node.right, indent + "    ", "no: "))
                stack.append((node.left, indent + "    ", "yes: "))
            lines.append(indent + branch + text)
        return "\n".join(lines)
