"""
Loss functions for GAN models.
"""
import keras.backend as K

from gan_models.common import Tensor


def _clip_log_prob(x: Tensor) -> Tensor:
    """Take a log on a probability; clip the value to avoid numerical issue of log(0)."""
    x_clipped = K.clip(x, K.epsilon(), 1.0 - K.epsilon())
    return K.log(x_clipped)


def gan_discriminator_loss(pred_real: Tensor, pred_generated: Tensor) -> Tensor:
    """Standard GAN loss function for discriminator.

    The function assumes the input tensors are raw discriminator output before sigmoid.

    Args:
        pred_real: Discriminator output on real input.
        pred_generated: Discriminator output on generated data.

    Returns:
        A scalar tensor for the loss.
    """
    return -K.mean(_clip_log_prob(K.sigmoid(pred_real)) + _clip_log_prob(1 - K.sigmoid(pred_generated)))


def gan_generator_loss(pred_generated: Tensor) -> Tensor:
    """Standard GAN loss for the generator.

    The function assumes the input tensors are raw discriminator output before sigmoid.

    Args:
        pred_generated: Discriminator output on generated data.

    Returns:
        A scalar tensor for the loss.
    """
    return -K.mean(_clip_log_prob(K.sigmoid(pred_generated)))
