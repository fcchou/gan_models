"""
Train the Yelp photo generator

Example command:
    python -m examples.yelp_photos.train_photo_generator --input-path=processed_photos.npy \
    --epochs=5 --batch-size=64 --iters=75
"""
import argparse

import numpy as np
import keras
import matplotlib.pyplot as plt

from gan_models.model_layers import get_dcgan_discriminator, get_dcgan_generator, ConvParams
from gan_models.gan_trainer import GanTrainer
from gan_models.common import plot_generated_images


PHOTO_SHAPE = (64, 64, 3)
G_INPUT_SIZE = 100


def get_dcgan_model():
    generator = get_dcgan_generator(
        input_dim=G_INPUT_SIZE,
        shape_layer1=(4, 4, 1024),
        conv_params_list=[
            ConvParams(filters=512, kernel_sizes=5, strides=2),
            ConvParams(filters=256, kernel_sizes=5, strides=2),
            ConvParams(filters=128, kernel_sizes=5, strides=2),
            ConvParams(filters=PHOTO_SHAPE[-1], kernel_sizes=5, strides=2),
        ],
        use_batch_norm=True,
    )
    discriminator = get_dcgan_discriminator(
        input_shape=PHOTO_SHAPE,
        conv_params_list=[
            ConvParams(filters=64, kernel_sizes=5, strides=2),
            ConvParams(filters=128, kernel_sizes=5, strides=2),
            ConvParams(filters=256, kernel_sizes=5, strides=2),
            ConvParams(filters=512, kernel_sizes=5, strides=2),
        ],
        use_batch_norm=False,
    )
    return generator, discriminator


def main():
    parser = argparse.ArgumentParser(description='Train GAN with Yelp photos')
    parser.add_argument('--input-path', required=True, help='Path to the preprocessed input data')
    parser.add_argument('--iters', type=int, required=True, help='Number of the training iterations')
    parser.add_argument('--epochs', type=int, required=True, help='Number of the epochs per iterations')
    parser.add_argument('--batch-size', type=int, required=True, help='Size of the training mini-batch')
    args = parser.parse_args()

    generator, discriminator = get_dcgan_model()

    model = GanTrainer(generator=generator, discriminator=discriminator)
    model.gan_compile(
        g_optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.9),
        d_optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.9),
        n_discriminator_training=5,
        is_wgan=True,
        wgan_gradient_lambda=10,
    )

    # Load the input and normalize
    training_image = np.load(args.input_path).astype(np.float32) / 255.0

    for i in range(args.iters):
        model.gan_fit(training_image, epochs=args.epochs, batch_size=args.batch_size)

        plot_generated_images(model.generator, G_INPUT_SIZE)
        plt.savefig('yelp_photo_gan{}.png'.format(i), dpi=600)

        model.generator.save('yelp_photo_g{}.hdf5'.format(i))
        model.discriminator.save('yelp_photo_d{}.hdf5'.format(i))


if __name__ == '__main__':
    main()
