"""Train and evaluate a scratch tree on a local CSV file."""

import argparse
import csv
import random

from .tree import DecisionTreeClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--target", required=True, help="Name of the class label column")
    parser.add_argument("--drop", nargs="*", default=[], help="Columns to exclude, e.g. identifiers")
    parser.add_argument("--categorical", nargs="*", default=[], help="Force columns to be categorical")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--max-leaf-nodes", type=int)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-tree", action="store_true")
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    with open(args.csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            parser.error("CSV headers must be unique")
        if args.target not in fields:
            parser.error("Target column does not exist")
        if any(c not in fields for c in args.drop + args.categorical):
            parser.error("Unknown --drop or --categorical column")
        data = list(reader)
    if len(data) < 2:
        parser.error("CSV needs at least two data rows")
    if any(None in row or any(v is None for v in row.values()) for row in data):
        parser.error("CSV rows must match the header width")
    if any(not row[args.target].strip() for row in data):
        parser.error("Target labels cannot be empty")
    names = [c for c in fields if c != args.target and c not in args.drop]
    if not names:
        parser.error("No feature columns remain")
    random.Random(args.seed).shuffle(data)
    n_test = min(len(data) - 1, max(1, round(len(data) * args.test_size)))
    train, test = data[n_test:], data[:n_test]
    # Infer types using training rows only. Empty feature cells become None.
    types = []
    for name in names:
        kind = "categorical" if name in args.categorical else "numeric"
        if kind == "numeric":
            try:
                for row in train:
                    if row[name].strip():
                        float(row[name])
            except ValueError:
                kind = "categorical"
        types.append(kind)

    def convert(partition):
        return [[None if not row[name].strip() else
                 float(row[name]) if kind == "numeric" else row[name]
                 for name, kind in zip(names, types)] for row in partition]

    try:
        model = DecisionTreeClassifier(max_depth=args.max_depth,
                                       max_leaf_nodes=args.max_leaf_nodes,
                                       min_samples_leaf=args.min_samples_leaf)
        model.fit(convert(train), [r[args.target] for r in train], names, types)
        accuracy = model.score(convert(test), [r[args.target] for r in test])
    except ValueError as exc:
        parser.error(str(exc) + "; use --categorical for nonnumeric or mixed columns")
    print(f"Training rows: {len(train)}; test rows: {len(test)}")
    print(f"Holdout accuracy: {accuracy:.4f}")
    print(f"Depth: {model.depth_}; leaves: {model.n_leaves_}; nodes: {model.node_count_}")
    if args.show_tree:
        print(model.export_text())


if __name__ == "__main__":
    main()
