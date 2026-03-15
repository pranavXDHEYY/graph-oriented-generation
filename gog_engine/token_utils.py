"""
token_utils.py - Accurate token counting for benchmark measurements.

Uses tiktoken (OpenAI's tokenizer) with the cl100k_base encoding, which is the
tokenizer used by GPT-4 and is a widely accepted proxy for token counts across
frontier models including Claude and Gemini. This replaces whitespace-split word
counting, which underestimates real token counts by 30-40% for code due to
punctuation, operators, and identifier fragmentation.

Why cl100k_base:
    - Open standard, reproducible by any reviewer
    - Conservative estimate relative to model-specific tokenizers
    - Consistent across cloud and local benchmark runs
"""

import os
from typing import Optional

# Lazy-loaded to avoid import cost at module level
_tokenizer = None

# Encoding used for all token estimates.
# cl100k_base is the GPT-4 / text-embedding-ada-002 tokenizer and a reasonable
# cross-model proxy. Swap to "o200k_base" for GPT-4o-family models if needed.
TIKTOKEN_ENCODING = "cl100k_base"


def _get_tokenizer():
    """Lazy-loads the tiktoken tokenizer on first use."""
    global _tokenizer
    if _tokenizer is None:
        try:
            import tiktoken
            _tokenizer = tiktoken.get_encoding(TIKTOKEN_ENCODING)
        except ImportError:
            raise ImportError(
                "tiktoken is required for accurate token counting. "
                "Install via: pip install tiktoken"
            )
    return _tokenizer


def count_tokens_in_string(text: str) -> int:
    """
    Returns the exact token count for a given string under cl100k_base encoding.

    Args:
        text: Any string — file contents, prompt text, or LLM response.

    Returns:
        Integer token count.
    """
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text))


def count_tokens_in_files(file_paths: list) -> int:
    """
    Returns the total token count across all provided file paths.

    Files that do not exist or cannot be read are silently skipped and do not
    contribute to the count. This matches the original benchmark behavior while
    producing accurate per-token measurements.

    Args:
        file_paths: List of absolute or relative file path strings.

    Returns:
        Integer total token count across all readable files.
    """
    tokenizer = _get_tokenizer()
    total = 0

    for path in file_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            total += len(tokenizer.encode(content))
        except Exception:
            # Non-fatal: skip unreadable files without breaking the benchmark run
            pass

    return total


def count_tokens_in_string_wordcount_fallback(text: str) -> int:
    """
    Word-count fallback for environments where tiktoken is unavailable.
    Retained for edge-case compatibility only — not used in benchmark runs.
    Produces estimates ~30-40% lower than actual token counts for code.
    """
    return len(text.split())


if __name__ == "__main__":
    # Sanity check: compare word-count vs tiktoken on a code sample
    sample = """
    import { defineStore } from 'pinia'
    import { authApi } from '@/services/authApi'

    export const useAuthStore = defineStore('auth', {
        state: () => ({
            user: null,
            isAuthenticated: false,
            role: 'guest',
        }),
    })
    """
    word_count = len(sample.split())
    token_count = count_tokens_in_string(sample)
    ratio = token_count / word_count if word_count > 0 else 0

    print(f"Sample code snippet:")
    print(f"  Word count (old method): {word_count}")
    print(f"  Token count (tiktoken):  {token_count}")
    print(f"  Ratio (tokens/words):    {ratio:.2f}x")
    print(f"\nFor code, tiktoken typically produces {ratio:.1f}x the word count.")
    print("Prior benchmarks using word-count underreported token load by this factor.")