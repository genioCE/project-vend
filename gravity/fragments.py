"""
Fragment taxonomy for the Gravity Model.

Every query decomposes into typed fragments. These categories map directly
to the capabilities of the tool pool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FragmentType(str, Enum):
    CONCEPT = "concept"
    ENTITY = "entity"
    TEMPORAL = "temporal"
    EMOTIONAL = "emotional"
    RELATIONAL = "relational"
    ARCHETYPAL = "archetypal"


@dataclass
class Fragment:
    type: FragmentType
    text: str
    embedding: np.ndarray | None = field(default=None, repr=False)


@dataclass
class DecompositionResult:
    fragments: list[Fragment]
    primary_mass_index: int
    claude_reasoning: str
    query_embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def primary_mass(self) -> Fragment:
        return self.fragments[self.primary_mass_index]
