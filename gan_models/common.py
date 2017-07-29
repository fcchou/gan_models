"""
Useful common utils for model training and visualization.
"""
import keras
import numpy as np
import matplotlib.pyplot as plt


def get_mnist_images():
    """Scale the MNIST images to (0, 1) and reshape it."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype('float32') / 255.0)[:, :, :, np.newaxis]
    x_test = (x_test.astype('float32') / 255.0)[:, :, :, np.newaxis]
    return (x_train, y_train), (x_test, y_test)


def mini_batch_generator(input_data, batch_size=64, shuffle=True):
    """Generator for training mini-batches

    Args:
        input_data (ndarray): Input training data.
        batch_size (int): Size of the mini batch.
        shuffle (bool): If the data is shuffled before mini batch generation.
    """
    n_samples = input_data.shape[0]
    n_batches = n_samples // batch_size
    if shuffle:
        shuffled_idx = np.arange(n_samples)
        np.random.shuffle(shuffled_idx)
        input_data = input_data[shuffled_idx]
    for j in range(n_batches):
        yield input_data[(j * batch_size):((j + 1) * batch_size)]


def plot_generated_images(generator, g_input_size):
    """Generate plots for images from generator.

    Args:
        generator (keras Model): GAN generator.
        g_input_size (int): Size of the generator input.
    """
    noise_input = np.random.normal(size=(49, g_input_size))

    generated_images = generator.predict(noise_input)
    image_shape = generated_images.shape
    if image_shape[-1] == 1:
        # If there is only one channel, flatten the channel dimension
        generated_images = generated_images[:, :, :, 0]

    plt.figure(figsize=(10, 10))
    for i in range(image_shape[0]):
        plt.subplot(7, 7, i + 1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
