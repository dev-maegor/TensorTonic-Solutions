import numpy as np

def gan_losses(real_probs: np.ndarray, fake_probs: np.ndarray) -> dict:
    """
    Returns discriminator_loss and generator_loss as Python floats.
    """
    real_probs = np.asarray(real_probs, dtype = np.float64)
    fake_probs = np.asarray(fake_probs, dtype = np.float64)
    eps = 1e-8
    real_probs = np.clip(real_probs, eps, 1.0 - eps)
    fake_probs = np.clip(fake_probs, eps, 1.0 - eps)
    dis_loss = - np.mean(np.log(real_probs) + np.log(1.0 - fake_probs))
    gen_loss = - np.mean(np.log(fake_probs))

    return {"discriminator_loss": float(dis_loss),"generator_loss": float(gen_loss)}
    pass