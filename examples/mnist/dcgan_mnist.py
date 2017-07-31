"""
Example script for training DCGAN on mnist dataset.

Example command:
     python -m examples.mnist.dcgan_mnist --epochs=150 --batch-size=64 --output-prefix=test --mode=wgan-gp

References:
    DCAGN: https://arxiv.org/abs/1511.06434
    WGAN: https://arxiv.org/abs/1701.07875
    Improved WGAN: https://arxiv.org/abs/1704.00028
"""
import argparse

import keras
import matplotlib.pyplot as plt

from gan_models.gan_trainer import GanTrainer
from gan_models.common import get_mnist_images, plot_generated_images
from .mnist_model import G_INPUT_SIZE, get_dcgan_model



def main():
    # Input Arguments
    parser = argparse.ArgumentParser(description='Train GAN model on MNIST dataset')
    parser.add_argument('--epochs', type=int, required=True, help='Number of the training epochs')
    parser.add_argument('--batch-size', type=int, required=True, help='Size of the training mini-batch')
    parser.add_argument('--output-prefix', type=str, required=True, help='String prefix of the output files')
    parser.add_argument(
        '--mode',
        choices=['gan', 'wgan', 'wgan-gp'],
        required=True,
        help='3 allowed modes. gan: original DCGAN; wgan: Wasserstein GAN; wgan-gp: Improved WGAN',
    )
    args = parser.parse_args()

    # Training parameters for different modes
    if args.mode == 'gan':
        batch_norm = True
        is_wgan = False
        n_discriminator_training = 1
        clip_weight = None
        wgan_gradient_lambda = 0
        g_optimizer = keras.optimizers.RMSprop(lr=0.0002, rho=0.5)
        d_optimizer = keras.optimizers.RMSprop(lr=0.0002, rho=0.5)
    elif args.mode == 'wgan':
        batch_norm = False
        is_wgan = True
        n_discriminator_training = 5
        clip_weight = 0.01
        wgan_gradient_lambda = 0
        g_optimizer = keras.optimizers.RMSprop(lr=0.0001)
        d_optimizer = keras.optimizers.RMSprop(lr=0.0001)
    else:
        batch_norm = False
        is_wgan = True
        n_discriminator_training = 5
        clip_weight = None
        wgan_gradient_lambda = 10
        g_optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0, beta_2=0.9)
        d_optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0, beta_2=0.9)


    generator, discriminator = get_dcgan_model(batch_norm=batch_norm, clip_weight=clip_weight)
    model = GanTrainer(generator=generator, discriminator=discriminator)
    model.gan_compile(
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        is_wgan=is_wgan,
        n_discriminator_training=n_discriminator_training,
        wgan_gradient_lambda=wgan_gradient_lambda,
    )

    (x_train, y_train), _ = get_mnist_images()
    model.gan_fit(x_train, epochs=args.epochs, batch_size=args.batch_size)

    plot_generated_images(model.generator, G_INPUT_SIZE)
    plt.savefig('{}.png'.format(args.output_prefix), dpi=200)

    model.generator.save('{}_g.hdf5'.format(args.output_prefix))
    model.discriminator.save('{}_d.hdf5'.format(args.output_prefix))


if __name__ == '__main__':
    main()
