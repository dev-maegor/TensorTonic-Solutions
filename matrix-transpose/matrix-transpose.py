import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    rows = len(A)
    cols = len(A[0])
    result = np.empty((cols, rows), dtype=np.array(A).dtype)
    for i in range(rows):
        for j in range(cols):
            result[j][i] = A[i][j]
    return result
    pass
