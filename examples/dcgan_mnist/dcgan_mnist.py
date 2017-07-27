"""
Example script for training DCGAN on mnist dataset.
"""
import tqdm
import keras
import numpy as np
import matplotlib.pyplot as plt

from gan_models.model_layers import get_dcgan_discriminator, get_dcgan_generator, ConvParams
from gan_models.loss_func import gan_discriminator_loss, gan_generator_loss
from gan_models.gan_trainer import GanTrainer


MNIST_SHAPE = (28, 28, 1)
G_INPUT_SIZE = 30


def get_gan_model():
    generator = get_dcgan_generator(
        input_dim=G_INPUT_SIZE,
        shape_layer1=(7, 7, 128),
        conv_params_list=[
            ConvParams(filters=64, kernel_sizes=5, strides=2),
            ConvParams(filters=MNIST_SHAPE[-1], kernel_sizes=5, strides=2),
        ],
    )
    discriminator = get_dcgan_discriminator(
        input_shape=MNIST_SHAPE,
        conv_params_list=[
            ConvParams(filters=64, kernel_sizes=5, strides=2),
            ConvParams(filters=128, kernel_sizes=5, strides=2),
        ],
    )

    model_train = GanTrainer(generator, discriminator)
    model_train.gan_compile(
        # Use RMSprop with the below parameters as DCGAN paper recommended
        keras.optimizers.RMSprop(lr=0.0002, rho=0.5),
        gan_generator_loss,
        gan_discriminator_loss,
    )
    return model_train


def get_training_images():
    """Scale the MNIST images to (-1, 1) and reshape it."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x = np.vstack((x_train, x_test))
    x = x.astype('float32')
    x_scaled = x / 255.0
    return x_scaled[:, :, :, None]


def train_gan_standard(gan_model, epochs, batch_size, input_data):
    """Standard GAN training protocol.

    We first train on a mini-batch on the discriminator, then update the generator -> loop
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
            d_loss_sum += gan_model.discriminator_train_on_batch(batch_input)
            g_loss_sum += gan_model.generator_train_on_batch(batch_input)
        print('Epoch {}, D-loss: {}, G-loss: {}'.format(i + 1, d_loss_sum / n_batches, g_loss_sum / n_batches))


def train_gan_fast(gan_model, epochs, batch_size, input_data):
    """Train GAN with the fast training trick, see the documentation of GanModel.

    On GTX 1070 is around 30% faster then the standard protocol above.
    """
    gan_model.gan_fit(input_data, epochs=epochs, batch_size=batch_size)


def plot_output(generator):
    noise_input = np.random.rand(49, G_INPUT_SIZE)
    generated_images = generator.predict(noise_input)

    plt.figure(figsize=(10, 10))
    for i in range(generated_images.shape[0]):
        plt.subplot(7, 7, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()


def main():
    input_images = get_training_images()

    gan_model = get_gan_model()

    # DCGAN recommends batch_size=128
    train_gan_standard(gan_model, epochs=10, batch_size=128, input_data=input_images)
    train_gan_fast(gan_model, epochs=150, batch_size=128, input_data=input_images)

    plot_output(gan_model.generator)
    plt.savefig('dcgan_mnist.png', dpi=200)

    gan_model.generator.save('dcgan_minst_g.hdf5')
    gan_model.discriminator.save('dcgan_minst_d.hdf5')


if __name__ == '__main__':
    main()
