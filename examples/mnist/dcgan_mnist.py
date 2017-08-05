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

import matplotlib.pyplot as plt

from gan_models.common import get_mnist_images, plot_generated_images
from .mnist_model import G_INPUT_SIZE, get_default_compiled_model


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

    model = get_default_compiled_model(args.mode)

    (x_train, y_train), _ = get_mnist_images()
    model.gan_fit(x_train, epochs=args.epochs, batch_size=args.batch_size)

    plot_generated_images(model.generator, G_INPUT_SIZE)
    plt.savefig('{}.png'.format(args.output_prefix), dpi=200)

    model.generator.save('{}_g.hdf5'.format(args.output_prefix))
    model.discriminator.save('{}_d.hdf5'.format(args.output_prefix))


if __name__ == '__main__':
    main()
