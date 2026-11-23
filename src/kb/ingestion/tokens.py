"""Token counting protocols and lightweight implementations."""

import math
from typing import Protocol


class TokenCounter(Protocol):
    """Protocol for counting tokens in text."""

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in the given text."""
        ...


class ApproxTokenCounter:
    """Lightweight token counter estimating tokens as ceil(len(text) / 4)."""

    def count_tokens(self, text: str) -> int:
        """Estimate token count based on character length."""
        if not text:
            return 0
        return math.ceil(len(text) / 4)
