"""
Example script for training DCGAN on mnist dataset.
This example uses the GanTrainerKeras class instead, without hacking keras internals.

Example command:
     python -m examples.mnist.dcgan_mnist_keras --epochs=150 --batch-size=64 --output-prefix=test

References:
    DCAGN: https://arxiv.org/abs/1511.06434
"""
import argparse
import time

import keras
import numpy as np
import matplotlib.pyplot as plt

from gan_models.gan_trainer_keras import GanTrainerKeras
from gan_models.common import get_mnist_images, plot_generated_images, mini_batch_generator
from .mnist_model import G_INPUT_SIZE, get_dcgan_model


def main():
    # Input Arguments
    parser = argparse.ArgumentParser(description='Train GAN model on MNIST dataset')
    parser.add_argument('--epochs', type=int, required=True, help='Number of the training epochs')
    parser.add_argument('--batch-size', type=int, required=True, help='Size of the training mini-batch')
    parser.add_argument('--output-prefix', type=str, required=True, help='String prefix of the output files')
    args = parser.parse_args()

    generator, discriminator = get_dcgan_model(batch_norm=False)
    trainer = GanTrainerKeras(generator=generator, discriminator=discriminator)
    trainer.compile(
        g_optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0, beta_2=0.9),
        d_optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0, beta_2=0.9),
        is_wgan=True,
        wgan_gradient_lambda=10,
    )

    (x_train, y_train), _ = get_mnist_images()

    count = 0
    for n in range(args.epochs):
        d_losses = []
        g_losses = []
        start_t = time.time()
        for x in mini_batch_generator(x_train, batch_size=args.batch_size):
            count += 1
            d_losses.append(trainer.discriminator_train_on_batch(x))
            if count % 5 == 0:
                g_losses.append(trainer.generator_train_on_batch(args.batch_size))
        print('Training time: {}, Epoch {}, D-loss: {}, G-loss: {}'.format(
            time.time() - start_t,
            n,
            np.mean(d_losses),
            np.mean(g_losses)
        ))

    plot_generated_images(trainer.generator, G_INPUT_SIZE)
    plt.savefig('{}.png'.format(args.output_prefix), dpi=200)

    trainer.generator.save('{}_g.hdf5'.format(args.output_prefix))
    trainer.discriminator.save('{}_d.hdf5'.format(args.output_prefix))


if __name__ == '__main__':
    main()
