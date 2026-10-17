import numpy as np

def generator(z: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Returns generated samples as a float64 array with shape (B, D).
    """
    z = np.asarray(z, dtype = np.float64)
    W = np.asarray(W, dtype = np.float64)
    b = np.asarray(b, dtype = np.float64)
    return np.tanh(z@W+b)
    pass