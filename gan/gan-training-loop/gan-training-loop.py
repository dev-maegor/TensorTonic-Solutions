import numpy as np

def train_discriminator_step(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    D_W: np.ndarray,
    learning_rate: float,
) -> dict:
    """
    Returns updated discriminator weights and the pre-update loss.
    """
    real_data = np.asarray(real_data, dtype = np.float64)
    fake_data = np.asarray(fake_data, dtype = np.float64)
    D_W = np.asarray(D_W, dtype = np.float64)
    real_probs =1/(1 + np.exp(-(real_data@D_W)))
    fake_probs =1/(1 + np.exp(-(fake_data@D_W)))
    eps = 1e-8
    real_clip = np.clip(real_probs, eps, 1 - eps)
    fake_clip = np.clip(fake_probs, eps, 1 - eps)
    dis_loss = -np.mean(np.log(real_clip) + np.log(1 - fake_clip))
    N = real_data.shape[0]
    grad = (
        real_data.T @ (real_probs - 1) + fake_data.T @ fake_probs
    )/N
    new_dw = D_W - learning_rate * grad
    return{"new_discriminator_weights": new_dw.astype(np.float64), "discriminator_loss": float(dis_loss)}
    pass