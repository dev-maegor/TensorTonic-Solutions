import numpy as np

def discriminator(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Returns discriminator probabilities as a float64 array with shape (B, 1).
    """
    x = np.asarray(x, dtype = np.float64)
    W = np.asarray(W, dtype = np.float64)
    res = x@W
    return 1/(1 + np.exp(-res))
    pass