"""Shared fixtures for graph-service tests.

When running locally (without spaCy installed), this conftest patches
``sys.modules`` so that ``import spacy`` and ``from spacy.tokens import Span``
succeed with lightweight mocks.  Inside Docker the real spaCy is available
and no patching occurs.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

try:
    import spacy  # noqa: F401

    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


def _install_spacy_stubs() -> None:
    """Insert minimal mock modules for spacy and spacy.tokens."""
    spacy_mod = types.ModuleType("spacy")
    spacy_mod.load = MagicMock(return_value=MagicMock(name="nlp"))  # type: ignore[attr-defined]

    tokens_mod = types.ModuleType("spacy.tokens")
    tokens_mod.Span = MagicMock(name="Span")  # type: ignore[attr-defined]

    sys.modules["spacy"] = spacy_mod
    sys.modules["spacy.tokens"] = tokens_mod


if not _SPACY_AVAILABLE:
    _install_spacy_stubs()
