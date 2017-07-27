"""
Example script for training DCGAN on mnist dataset.
See https://arxiv.org/abs/1511.06434 for DCGAN paper
"""
import tqdm
import keras
import numpy as np
import matplotlib.pyplot as plt

from gan_models.model_layers import get_dcgan_discriminator, get_dcgan_generator, ConvParams
from gan_models.gan_trainer import GanTrainer
from gan_models.common import get_mnist_images, train_gan_standard


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
    )
    return model_train


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
    input_images, _ = get_mnist_images()

    gan_model = get_gan_model()

    # DCGAN recommends batch_size=128
    for epoch_num, d_loss, g_loss in train_gan_standard(
        input_images,
        lambda x: gan_model.discriminator_train_on_batch(x),
        lambda x: gan_model.generator_train_on_batch(x),
        epochs=5,
        batch_size=128,
    ):
        print('Epoch {}; D-loss: {}, G-loss: {}'.format(epoch_num, d_loss, g_loss))

    # Train GAN with the fast training trick, see the documentation of GanModel.
    # On GTX 1070 is around 30% faster then the standard protocol above.
    gan_model.gan_fit(input_images, epochs=100, batch_size=128)

    plot_output(gan_model.generator)
    plt.savefig('dcgan_mnist.png', dpi=200)

    gan_model.generator.save('dcgan_minst_g.hdf5')
    gan_model.discriminator.save('dcgan_minst_d.hdf5')


if __name__ == '__main__':
    main()
