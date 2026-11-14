"""Strict source-specific formatters for the K2 pretraining/SFT subsets.

The checked release exposes preformatted `text` in every subset listed in K2.
Preserve it verbatim, including upstream role markers, reasoning and indentation.
Never replace a missing document with metadata or a guessed field combination.
"""
from collections.abc import Mapping

from .sources import K2


def _text_formatter(repo, subset):
    def format_subset(row):
        if not isinstance(row, Mapping):
            return None
        value = row.get('text')
        if not isinstance(value, str) or not value.strip():
            return None
        return value
    format_subset.__name__ = (repo.split('/')[-1] + '_' + subset).replace('-', '_')
    format_subset.__doc__ = f'Preserve the verified text field from {repo}/{subset}; reject broken rows.'
    return format_subset


K2_FORMATTERS = {(repo, subset): _text_formatter(repo, subset)
                 for repo, (_, subsets) in K2.items() for subset in subsets}


def get_k2_formatter(repo, subset):
    """Return a registered subset formatter; unknown subsets require explicit support."""
    try:
        return K2_FORMATTERS[repo, subset]
    except KeyError as e:
        raise ValueError(f'No verified K2 formatter for {repo}/{subset}') from e
