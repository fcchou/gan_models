from gan_models.model_layers import UniformNoiseGenerator, get_dcgan_discriminator, get_dcgan_generator, ConvParams
from gan_models.model_layers import binary_log_loss_constant
import keras


MNIST_SHAPE = (28, 28, 1)
RANDOM_INPUT_SIZE = 30

def get_models():
    input_tensor =  keras.Input(shape=MNIST_SHAPE)
    random_generator = UniformNoiseGenerator(-1, 1, RANDOM_INPUT_SIZE)
    random_input = random_generator(input_tensor)
    generator = get_dcgan_generator(
        input_dim=RANDOM_INPUT_SIZE,
        shape_layer1=(7, 7, 256),
        conv_params_list=[
            ConvParams(filters=128, kernel_sizes=5, strides=2),
            ConvParams(filters=MNIST_SHAPE[-1], kernel_sizes=5, strides=2),
        ],
    )
    discriminator = get_dcgan_discriminator(
        input_shape=MNIST_SHAPE,
        conv_params_list=[
            ConvParams(filters=256, kernel_sizes=3, strides=1),
            ConvParams(filters=128, kernel_sizes=3, strides=1),
            ConvParams(filters=64, kernel_sizes=3, strides=1),
        ],
    )
    true_image_output = discriminator(input_tensor)
    generated_train_images = generator(random_input)
    generated_image_output = discriminator(generated_train_images)
    train_model = keras.models.Model(inputs=[input_tensor], outputs=[true_image_output, generated_image_output])
    train_model.compile(
        # Use RMSprop with the following configuration as recommended in DCGAN paper
        optimizer=keras.optimizers.RMSprop(lr=0.0002, rho=0.5),
        loss=[binary_log_loss_constant(True), binary_log_loss_constant(False)],
    )
    return train_model, generator


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    train_model, generator = get_models()
    train_model.fit(x=x_train, epochs=5)


if __name__ == '__main__':
    main()
