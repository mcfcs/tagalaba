"""tagalaba: a natural-sounding Tagalog crossword clue generator."""

from .baseline import Candidate, generate
from .sources import ClueIndex, WordResources, get_index

__all__ = ["Candidate", "generate", "ClueIndex", "WordResources", "get_index"]
