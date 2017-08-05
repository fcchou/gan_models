import keras

from gan_models.gan_trainer import GanTrainer
from gan_models.model_layers import get_dcgan_discriminator, get_dcgan_generator, ConvParams

MNIST_SHAPE = (28, 28, 1)
G_INPUT_SIZE = 100


def get_dcgan_model(batch_norm=True, clip_weight=None):
    generator = get_dcgan_generator(
        input_dim=G_INPUT_SIZE,
        shape_layer1=(7, 7, 128),
        conv_params_list=[
            ConvParams(filters=64, kernel_sizes=5, strides=2),
            ConvParams(filters=MNIST_SHAPE[-1], kernel_sizes=5, strides=2),
        ],
        use_batch_norm=batch_norm,
    )
    discriminator = get_dcgan_discriminator(
        input_shape=MNIST_SHAPE,
        conv_params_list=[
            ConvParams(filters=64, kernel_sizes=5, strides=2),
            ConvParams(filters=128, kernel_sizes=5, strides=2),
        ],
        use_batch_norm=batch_norm,
        clip_weight=clip_weight,
    )
    return generator, discriminator


def get_default_compiled_model(mode):
    # Training parameters for different modes
    if mode == 'gan':
        batch_norm = True
        is_wgan = False
        n_discriminator_training = 1
        clip_weight = None
        wgan_gradient_lambda = 0
        g_optimizer = keras.optimizers.RMSprop(lr=0.0002, rho=0.5)
        d_optimizer = keras.optimizers.RMSprop(lr=0.0002, rho=0.5)
    elif mode == 'wgan':
        batch_norm = False
        is_wgan = True
        n_discriminator_training = 5
        clip_weight = 0.01
        wgan_gradient_lambda = 0
        g_optimizer = keras.optimizers.RMSprop(lr=0.0001)
        d_optimizer = keras.optimizers.RMSprop(lr=0.0001)
    elif mode == 'wgan-gp':
        batch_norm = False
        is_wgan = True
        n_discriminator_training = 5
        clip_weight = None
        wgan_gradient_lambda = 10
        g_optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0, beta_2=0.9)
        d_optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0, beta_2=0.9)
    else:
        raise ValueError('Invalid input mode')

    generator, discriminator = get_dcgan_model(batch_norm=batch_norm, clip_weight=clip_weight)
    model = GanTrainer(generator=generator, discriminator=discriminator)
    model.gan_compile(
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        is_wgan=is_wgan,
        n_discriminator_training=n_discriminator_training,
        wgan_gradient_lambda=wgan_gradient_lambda,
    )
    return model
