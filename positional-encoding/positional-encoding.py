import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    # Write code here
    positions = np.arange(seq_len, dtype=float).reshape(-1,1)
    num_freqs = (d_model + 1)//2
    i = np.arange(num_freqs, dtype = float)
    div_term = np.power(base, (2*i)/d_model)
    angles = positions/div_term
    pe = np.empty((seq_len, d_model), dtype = float)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :d_model//2])
    return pe
    pass