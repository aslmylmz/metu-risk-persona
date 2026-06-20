"""Ensure the repository root is importable so ``import scoring`` works under pytest."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
