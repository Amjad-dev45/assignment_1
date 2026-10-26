"""A dependency-free, information-gain decision tree classifier."""

from .tree import DecisionTreeClassifier, entropy

__all__ = ["DecisionTreeClassifier", "entropy"]
__version__ = "0.1.0"
