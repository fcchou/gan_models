from typing import Callable

import keras
import numpy as np
import tqdm
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    Tensor = tf.Tensor
except ImportError:
    Tensor = object


def get_mnist_images():
    """Scale the MNIST images to (0, 1) and reshape it."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype('float32') / 255.0)[:, :, :, np.newaxis]
    x_test = (x_test.astype('float32') / 255.0)[:, :, :, np.newaxis]
    return (x_train, y_train), (x_test, y_test)


def train_gan_standard(
    input_data: np.ndarray,
    train_d: Callable[[np.ndarray], float],
    train_g: Callable[[np.ndarray], float],
    epochs: int=10,
    batch_size: int=64,
):
    """Standard GAN training protocol.

    We first train on a mini-batch on the discriminator, then update the generator -> loop

    Args:
        input_data: Input training data.
        train_d: Function that train discriminator with a mini-batch, returns the loss value.
        train_g: Function that train generator with a mini-batch, returns the loss value.
        epochs: Training epochs.
        batch_size: Size of the mini-batch.

    Yields:
        For each epoch, (epoch number, avg discriminator loss, avg generator loss)
    """
    n_samples = input_data.shape[0]
    n_batches = n_samples // batch_size
    for i in range(epochs):
        shuffled_idx = np.arange(n_samples)
        np.random.shuffle(shuffled_idx)
        shuffled_input = input_data[shuffled_idx]
        d_loss_sum = 0
        g_loss_sum = 0
        for j in tqdm.trange(n_batches):
            batch_input = shuffled_input[(j * batch_size):((j + 1) * batch_size)]
            d_loss_sum += train_d(batch_input)
            d_loss_sum += train_g(batch_input)
        yield i, d_loss_sum / n_batches, g_loss_sum / n_batches


def plot_output(generator, g_input_size):
    """Generate plots from generator.

    Args:
        generator: GAN generator.
        g_input_size: Size of the generator input.
    """
    noise_input = np.random.normal(size=(49, g_input_size))

    generated_images = generator.predict(noise_input)

    plt.figure(figsize=(10, 10))
    for i in range(generated_images.shape[0]):
        plt.subplot(7, 7, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
