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
