import math
import random
import unittest

from entropy_tree import DecisionTreeClassifier, entropy


class TreeTests(unittest.TestCase):
    def test_entropy_known_values(self):
        self.assertEqual(entropy([]), 0)
        self.assertEqual(entropy(["a"] * 4), 0)
        self.assertEqual(entropy([0, 1]), 1)
        self.assertAlmostEqual(entropy([0, 0, 0, 1]), 0.8112781244591328)

    def test_perfect_numeric_split_and_gain(self):
        model = DecisionTreeClassifier().fit([[1], [2], [3], [4]], [0, 0, 1, 1])
        self.assertEqual(model.predict([[1], [4]]), [0, 1])
        self.assertEqual(model.root_.value, 2)
        self.assertAlmostEqual(model.root_.gain, 1)
        self.assertEqual(model.depth_, 1)

    def test_numeric_gain_against_brute_force(self):
        rng = random.Random(123)
        for _ in range(30):
            X = [[rng.randrange(8), rng.randrange(5)] for _ in range(25)]
            y = [rng.randrange(3) for _ in X]
            gains = [0.0]
            for j in range(2):
                for threshold in sorted({r[j] for r in X})[:-1]:
                    left = [label for row, label in zip(X, y) if row[j] <= threshold]
                    right = [label for row, label in zip(X, y) if row[j] > threshold]
                    gains.append(entropy(y) - (len(left) * entropy(left) +
                                 len(right) * entropy(right)) / len(y))
            model = DecisionTreeClassifier(max_depth=1).fit(X, y)
            self.assertAlmostEqual(model.root_.gain, max(gains))

    def test_categorical_multiclass_and_unseen_values(self):
        X = [["red"], ["green"], ["blue"]] * 3
        y = ["r", "g", "b"] * 3
        model = DecisionTreeClassifier().fit(X, y)
        self.assertEqual(model.predict(X), y)
        self.assertIn(model.predict([["purple"]])[0], model.classes_)
        self.assertEqual(model.predict_proba([["red"]]), [[1, 0, 0]])

    def test_missing_values_and_missing_only_split(self):
        for missing in (None, float("nan")):
            model = DecisionTreeClassifier().fit([[1], [1], [missing], [missing]], [0, 0, 1, 1])
            self.assertEqual(model.predict([[1], [missing]]), [0, 1])
        model = DecisionTreeClassifier().fit([[0], [1], [2], [None]], [0, 0, 1, 1])
        self.assertEqual(model.predict([[None], [0], [2]]), [1, 0, 1])

    def test_all_missing_constant_and_tie(self):
        for X in ([[None]] * 4, [[3]] * 4):
            model = DecisionTreeClassifier().fit(X, ["b", "a", "a", "b"])
            self.assertEqual(model.n_leaves_, 1)
            self.assertEqual(model.predict(X), ["b"] * 4)
            self.assertEqual(model.predict_proba(X)[0], [0.5, 0.5])

    def test_size_constraints(self):
        X, y = [[i] for i in range(30)], [i % 3 for i in range(30)]
        for depth in range(4):
            model = DecisionTreeClassifier(max_depth=depth).fit(X, y)
            self.assertLessEqual(model.depth_, depth)
        for leaves in range(1, 8):
            model = DecisionTreeClassifier(max_depth=None, max_leaf_nodes=leaves).fit(X, y)
            self.assertLessEqual(model.n_leaves_, leaves)
            self.assertEqual(model.node_count_, 2 * model.n_leaves_ - 1)

    def test_minimum_samples_and_gain(self):
        X, y = [[i] for i in range(12)], [i % 2 for i in range(12)]
        model = DecisionTreeClassifier(min_samples_leaf=3).fit(X, y)
        stack = [model.root_]
        while stack:
            node = stack.pop()
            if node.is_leaf:
                self.assertGreaterEqual(node.n_samples, 3)
            else:
                stack.extend([node.left, node.right])
        self.assertEqual(DecisionTreeClassifier(min_samples_split=20).fit(X, y).node_count_, 1)
        self.assertEqual(DecisionTreeClassifier(min_gain=1.1).fit(X, y).node_count_, 1)

    def test_extreme_numeric_values(self):
        X = [[-1e308], [1e308]]
        self.assertEqual(DecisionTreeClassifier().fit(X, [0, 1]).predict(X), [0, 1])

    def test_mixed_features_and_category_override(self):
        X = [[0, "red"], [0, "blue"], [1, "red"], [1, "blue"]]
        model = DecisionTreeClassifier().fit(X, [0, 1, 0, 1], feature_types=["categorical", "categorical"])
        self.assertEqual(model.score(X, [0, 1, 0, 1]), 1)

    def test_display(self):
        model = DecisionTreeClassifier().fit([[1], [2]], ["a", "b"], ["height"])
        text = model.export_text()
        for expected in ("height <=", "yes:", "no:", "predict", "gain=1.0000"):
            self.assertIn(expected, text)

    def test_prediction_validation(self):
        model = DecisionTreeClassifier()
        with self.assertRaises(RuntimeError):
            model.predict([[0]])
        model.fit([[0], [1]], [0, 1])
        for X in ([[0, 1]], [["invalid"]], [[math.inf]]):
            with self.assertRaises(ValueError):
                model.predict(X)
        self.assertEqual(model.predict([]), [])
        with self.assertRaises(ValueError):
            model.score([], [])

    def test_invalid_training_data(self):
        for X, y in [([], []), ([[]], [1]), ([[1], [2]], [0]),
                     ([[1], [1, 2]], [0, 1]), ([[math.inf]], [0]),
                     ([[1]], [None]), ([[1]], [math.nan]), ([[{}]], [0]),
                     ([1, 2], [0, 1])]:
            with self.subTest(X=X, y=y), self.assertRaises(ValueError):
                DecisionTreeClassifier().fit(X, y)
        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit([[1]], [0], feature_names=[])
        with self.assertRaises(ValueError):
            DecisionTreeClassifier().fit([[1]], [0], feature_types=["bad"])

    def test_invalid_hyperparameters(self):
        for params in ({"max_depth": -1}, {"max_depth": True}, {"max_leaf_nodes": 0},
                       {"min_samples_leaf": 0}, {"min_samples_split": 1},
                       {"min_gain": -1}, {"min_gain": math.nan}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                DecisionTreeClassifier(**params)

    def test_refit_and_reproducibility(self):
        X, y = [[0], [1], [2], [3]], ["a", "b", "b", "a"]
        a = DecisionTreeClassifier().fit(X, y)
        b = DecisionTreeClassifier().fit(X, y)
        self.assertEqual(a.export_text(), b.export_text())
        a.fit([[0], [1]], ["new", "new"])
        self.assertEqual(a.classes_, ["new"])
        self.assertEqual(a.node_count_, 1)

    def test_zero_gain_stops(self):
        # Greedy positive-gain trees cannot discover the XOR interaction.
        model = DecisionTreeClassifier().fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0])
        self.assertEqual(model.node_count_, 1)


if __name__ == "__main__":
    unittest.main()
