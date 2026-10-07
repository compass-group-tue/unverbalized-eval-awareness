"""Synthetic-document generation for the model-organism experiments.

The package is intentionally independent of training and evaluation.  It reads
a universe-context JSONL file and writes synthetic-document artifacts that
downstream stages may consume.
"""

from .schemas import SynthDocument, UniverseContext

__all__ = ["SynthDocument", "UniverseContext"]
