"""
Example script for training DCGAN on mnist dataset.

References: https://arxiv.org/abs/1511.06434
"""
import keras
import matplotlib.pyplot as plt

from gan_models.gan_trainer import GanTrainer
from gan_models.common import get_mnist_images, plot_generated_images
from .mnist_model import G_INPUT_SIZE, get_dcgan_model



def main():
    (x_train, y_train), _ = get_mnist_images()

    generator, discriminator = get_dcgan_model(batch_norm=True)
    model = GanTrainer(generator=generator, discriminator=discriminator)
    model.gan_compile(
        # RMSprop and the following config following the paper parameters
        g_optimizer=keras.optimizers.RMSprop(lr=0.0002, rho=0.5),
        d_optimizer=keras.optimizers.RMSprop(lr=0.0002, rho=0.5),
    )

    model.gan_fit(x_train, epochs=150, batch_size=128)

    plot_generated_images(model.generator, G_INPUT_SIZE)
    plt.savefig('dcgan_mnist.png', dpi=200)

    model.generator.save('dcgan_minst_g.hdf5')
    model.discriminator.save('dcgan_minst_d.hdf5')


if __name__ == '__main__':
    main()
