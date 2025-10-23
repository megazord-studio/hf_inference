"""Data conversion utilities for structured data.

Provides utilities for converting various table-like structures to pandas DataFrames.
"""

from typing import List

import pandas as pd


def to_dataframe(table_like: List[List[str]]) -> pd.DataFrame:
    """
    Convert a list-of-lists table structure to a pandas DataFrame.

    Treats the first row as headers if all elements are strings.
    Otherwise, creates a DataFrame with default integer column names.

    Args:
        table_like: Nested list representing a table

    Returns:
        pandas DataFrame with appropriate headers

    Example:
        >>> table = [["city", "country"], ["Bern", "Switzerland"], ["Paris", "France"]]
        >>> df = to_dataframe(table)
        >>> df.columns.tolist()
        ['city', 'country']

        >>> # No header case
        >>> table = [[1, 2], [3, 4]]
        >>> df = to_dataframe(table)
        >>> df.shape
        (2, 2)
    """
    rows = [[str(x) for x in r] for r in table_like]
    if rows and all(isinstance(c, str) for c in rows[0]):
        header = rows[0]
        data = rows[1:] if len(rows) > 1 else []
        return pd.DataFrame(data, columns=header)
    return pd.DataFrame(rows)


__all__ = ["to_dataframe"]
