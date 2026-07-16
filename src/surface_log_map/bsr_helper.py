"""Helpers for constructing block sparse row matrices."""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import bsr_matrix


def build_bsr_matrix(
    data_ij: tuple[NDArray[np.float64], NDArray[np.int_]],
    shape: tuple[int, int],
    blocksize: tuple[int, int] | None = None,
) -> bsr_matrix:
    """Build a BSR matrix from dense blocks and their block coordinates.

    Parameters
    ----------
    data_ij
        ``(data, ij)``, where ``data`` is a float array with shape
        ``(number_of_blocks, R, C)`` and ``ij`` is an integer array with shape
        ``(2, number_of_blocks)``. ``data[k]`` is stored at block-grid
        coordinate ``(ij[0, k], ij[1, k])``.
    shape
        Scalar matrix shape ``(M, N)``. ``M`` must be divisible by ``R`` and
        ``N`` must be divisible by ``C``.
    blocksize
        Block shape ``(R, C)``. If omitted, it is inferred from ``data``.

    Returns
    -------
    scipy.sparse.bsr_matrix
        A 2-D sparse matrix with scalar shape ``(M, N)`` and block size
        ``(R, C)``.

    Notes
    -----
    Each block coordinate must occur at most once. BSR is efficient when the
    sparsity pattern is built in one pass; adding new blocks afterward is not.

    Examples
    --------
    >>> blocks = np.array([np.eye(3), 2 * np.eye(3)], dtype=float)
    >>> ij = np.array([[0, 1], [1, 0]], dtype=int)
    >>> matrix = build_bsr_matrix(
    ...     (blocks, ij), shape=(6, 6), blocksize=(3, 3)
    ... )
    >>> matrix.shape
    (6, 6)
    >>> np.array_equal(matrix.data[0], np.eye(3))
    True
    """
    data, ij = data_ij
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")
    if not isinstance(ij, np.ndarray):
        raise TypeError("ij must be a numpy.ndarray")
    if data.dtype != np.dtype(float):
        raise TypeError("data must have dtype=float")
    if ij.dtype != np.dtype(int):
        raise TypeError("ij must have dtype=int")
    if ij.ndim != 2 or ij.shape[0] != 2:
        raise ValueError("ij must have shape (2, number_of_blocks)")

    rows = ij[0]
    columns = ij[1]

    if len(shape) != 2 or shape[0] < 0 or shape[1] < 0:
        raise ValueError("shape=(M, N) must contain two nonnegative dimensions")
    if data.ndim != 3:
        raise ValueError("data must have shape (number_of_blocks, rows, columns)")
    if not (len(data) == len(rows) == len(columns)):
        raise ValueError("data, rows, and columns must have the same length")

    inferred_blocksize = data.shape[1:]
    if blocksize is None:
        blocksize = inferred_blocksize
    if (
        len(blocksize) != 2
        or blocksize[0] <= 0
        or blocksize[1] <= 0
    ):
        raise ValueError("blocksize must contain two positive dimensions")
    if tuple(blocksize) != inferred_blocksize:
        raise ValueError(
            f"data blocks have shape {inferred_blocksize}, not {tuple(blocksize)}"
        )

    M, N = shape
    R, C = blocksize
    if M % R != 0 or N % C != 0:
        raise ValueError(
            f"blocksize {(R, C)} must evenly divide matrix shape {(M, N)}"
        )
    number_of_block_rows = M // R
    number_of_block_columns = N // C

    if np.any(rows < 0) or np.any(rows >= number_of_block_rows):
        raise IndexError("block row index must be in [0, M / R)")
    if np.any(columns < 0) or np.any(columns >= number_of_block_columns):
        raise IndexError("block column index must be in [0, N / C)")

    # Standard BSR storage groups entries by block row, with block-column
    # indices sorted inside each row.
    order = np.lexsort((columns, rows)) # sort into (0,0), (0,1), (2,0), (2,4), ...
    rows = rows[order].astype(np.intp, copy=False)
    columns = columns[order].astype(np.intp, copy=False)
    data = data[order]

    # check for duplicate block coordinates (i,j), which are not allowed in BSR
    if len(rows) > 1:
        duplicate = (rows[1:] == rows[:-1]) & (columns[1:] == columns[:-1])
        if np.any(duplicate):
            position = np.flatnonzero(duplicate)[0] + 1
            coordinate = (int(rows[position]), int(columns[position]))
            raise ValueError(f"duplicate block coordinate {coordinate}")

    counts = np.bincount(rows, minlength=number_of_block_rows)
    indptr = np.concatenate(([0], np.cumsum(counts, dtype=np.intp)))
    return bsr_matrix((data, columns, indptr), shape=(M, N))
