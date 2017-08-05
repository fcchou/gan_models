"""
Example script for training MLP-based GAN on mnist dataset (using improved WGAN).

Example command:
     python -m examples.mnist.mlp_mnist --epochs=150 --batch-size=64 --output-prefix=test
"""
import argparse

import keras
import matplotlib.pyplot as plt

from gan_models.gan_trainer import GanTrainer
from gan_models.common import get_mnist_images, plot_generated_images
from .mnist_model import G_INPUT_SIZE, MNIST_SHAPE


def get_gan_mlp_model():
    generator = keras.models.Sequential([
        keras.layers.Dense(512, input_dim=G_INPUT_SIZE, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(MNIST_SHAPE[0] * MNIST_SHAPE[1] * MNIST_SHAPE[2], activation='sigmoid'),
        keras.layers.Reshape(MNIST_SHAPE),
    ])
    discriminator = keras.models.Sequential([
        keras.layers.Flatten(input_shape=MNIST_SHAPE),
        keras.layers.Dense(512),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dense(512),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dense(512),
        keras.layers.LeakyReLU(0.2),
        keras.layers.Dense(1),
    ])
    return generator, discriminator


def main():
    # Input Arguments
    parser = argparse.ArgumentParser(description='Train GAN model on MNIST dataset')
    parser.add_argument('--epochs', type=int, required=True, help='Number of the training epochs')
    parser.add_argument('--batch-size', type=int, required=True, help='Size of the training mini-batch')
    parser.add_argument('--output-prefix', type=str, required=True, help='String prefix of the output files')
    args = parser.parse_args()

    generator, discriminator = get_gan_mlp_model()
    model = GanTrainer(generator=generator, discriminator=discriminator)
    model.gan_compile(
        g_optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.9),
        d_optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.9),
        is_wgan=True,
        wgan_gradient_lambda=10,
        n_discriminator_training=5,
    )

    (x_train, y_train), _ = get_mnist_images()

    model.gan_fit(x_train, epochs=args.epochs, batch_size=args.batch_size)

    plot_generated_images(model.generator, G_INPUT_SIZE)
    plt.savefig('{}.png'.format(args.output_prefix), dpi=200)

    model.generator.save('{}_g.hdf5'.format(args.output_prefix))
    model.discriminator.save('{}_d.hdf5'.format(args.output_prefix))


if __name__ == '__main__':
    main()
