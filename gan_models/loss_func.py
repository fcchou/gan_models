"""
Loss functions for GAN models.
"""
import keras.backend as K


def clip_log_prob(x):
    """Take a log on probabilities; clip the value to avoid numerical issue of log(0)."""
    x_clipped = K.clip(x, K.epsilon(), 1.0 - K.epsilon())
    return K.log(x_clipped)


def gan_discriminator_loss(pred_real, pred_generated):
    """Standard GAN loss function for discriminator.

    The function assumes the input tensors are raw discriminator output before sigmoid.

    Args:
        pred_real (Tensor): Discriminator output on real input.
        pred_generated (Tensor): Discriminator output on generated data.

    Returns:
        A 1D tensor for the loss (shape=(batch_size,)).
    """
    return -K.mean(clip_log_prob(K.sigmoid(pred_real)) + clip_log_prob(1 - K.sigmoid(pred_generated)), axis=-1)


def gan_generator_loss(pred_generated):
    """Standard GAN loss for the generator.

    The function assumes the input tensors are raw discriminator output before sigmoid.

    Args:
        pred_generated: Discriminator output on generated data.

    Returns:
        A 1D tensor for the loss (shape=(batch_size,)).
    """
    return -K.mean(clip_log_prob(K.sigmoid(pred_generated)), axis=-1)


def get_wgan_gradient_penalty(x_real, x_gen, discriminator):
    """Gradient penalty for improved WGAN.

    Args:
        x_real (Tensor): Input tensor of real data.
        x_gen (Tensor): Tensor for generated data.
        discriminator (keras Model): Discriminator for WGAN.

    Returns:
        A 1D tensor for the gradient penalty (shape=(batch_size,)).

    References:
        See improved WGAN paper: https://arxiv.org/abs/1704.00028.
    """
    batch_size = K.shape(x_real)[0]

    epsilon_shape = [batch_size] + [1] * (K.ndim(x_real) - 1)
    epsilon = K.random_uniform(epsilon_shape)
    x_interpolate = epsilon * x_real + (1 - epsilon) * x_gen

    d_output = discriminator(x_interpolate)
    grad = K.gradients(d_output, x_interpolate)
    grad_reshape = K.reshape(grad, (batch_size, -1))
    grad_norm = K.sqrt(K.sum(K.square(grad_reshape), axis=1))
    return K.square(grad_norm - 1)


def identity_loss(y_true, y_pred):
    """A dummy keras loss function for training unsupervised model.

    The loss is just the value of y_pred. To use it, compute your desired loss as the model output,
    and compile your keras model with this identity_loss.

    Args:
        y_true (Tensor):
            True prediction in supervised learning. Not used here,
            but need to include it to be compatible with keras API.
        y_pred (Tensor): The model prediction

    Returns:
        Returns y_pred as an 1D tensor.
    """
    return y_pred[:, 0]