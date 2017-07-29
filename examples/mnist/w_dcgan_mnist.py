"""
"""
import keras
import matplotlib.pyplot as plt

from gan_models.gan_trainer import GanTrainer
from gan_models.common import get_mnist_images, plot_generated_images
from .mnist_model import G_INPUT_SIZE, get_dcgan_model


def main():
    (x_train, y_train), _ = get_mnist_images()

    generator, discriminator = get_dcgan_model(batch_norm=False, clip_weight=0.01)
    model = GanTrainer(generator=generator, discriminator=discriminator)
    model.gan_compile(
        g_optimizer=keras.optimizers.RMSprop(lr=0.0001),
        d_optimizer=keras.optimizers.RMSprop(lr=0.0001),
        is_wgan=True,
        n_discriminator_training=5,
    )

    model.gan_fit(x_train, epochs=300, batch_size=64)

    plot_generated_images(model.generator, G_INPUT_SIZE)
    plt.savefig('w_dcgan_mnist.png', dpi=200)

    model.generator.save('w_dcgan_minst_g.hdf5')
    model.discriminator.save('w_dcgan_minst_d.hdf5')


if __name__ == '__main__':
    main()
