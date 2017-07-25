from collections import namedtuple
from typing import Tuple, Sequence, Callable, Any

from keras.models import Sequential
from keras import layers
import keras.backend as K


def _clip_log_prob(x):
    x_clipped = K.clip(x, K.epsilon(), 1.0 - K.epsilon())
    return K.log(x_clipped)

def gan_discriminator_loss(pred_real, pred_generated):
    return -K.mean(_clip_log_prob(K.sigmoid(pred_real)) + _clip_log_prob(1 - K.sigmoid(pred_generated)))

def gan_generator_loss(pred_generated):
    return -K.mean(_clip_log_prob(K.sigmoid(pred_generated)))


def _gan_discriminator_loss(y_true, y_pred):
    """Standard GAN loss function for discriminator in Keras form.

    Args:
        y_true: Ignored, as in this is unsupervised.
        y_pred:
            Model prediction tensor, shape=(batch_size, 2).
            The 1st column is the probability predicted on true images;
            the 2nd column is the probability predicted on fake images.

    Returns:
        The computed loss.
    """
    return -_clip_log_prob(K.sigmoid(pred_real)) -_clip_log_prob(1 - K.sigmoid(pred_generated))


def _gan_generator_loss(y_true, y_pred):
    """Standard GAN loss for the generator in Keras form.

    Args:
        y_true: Ignored, as in this is unsupervised.
        y_pred: Tensor of shape (batch_size,). The predicted probability of the generated images.

    Returns:
        The computed loss.
    """
    return _binary_log_loss_constant(y_pred, True)
