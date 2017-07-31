"""
Another model class to help training GAN.

This trainer class reuses most of the existing keras training utils without hacking into the internals;
you need train generator and discriminator individually with mini-batch.

See also gan_trainer.py for a faster trainer that hacks keras internals.
"""
import numpy as np
from keras.models import Model, Input, Sequential
from keras import layers
import keras.backend as K

from gan_models import loss_func


class GanTrainerKeras(object):

    def __init__(self, generator, discriminator):
        """Helper object for training GAN model.

        Args:
            generator (keras Model):
                A keras model for GAN generator.
                The input should be a 1D noise vector of Uniform(-1, 1).
                The output shape should match the input shape of discriminator
            discriminator (keras Model):
                A keras model for GAN discriminator.
                The discriminator should return the single raw value of the last layer (before e.g. sigmoid).

        Examples:
            >>> model = GanTrainerKeras(generator=generator, discriminator=discriminator)
            >>> model.compile(g_optimizer='rmsprop', d_optimizer='rmsprop')
            >>> model.discriminator_train_on_batch(x_train_batch)  # Train the discriminator with a mini-batch
            >>> model.generator_train_on_batch(64)  # Train the generator with batch_size=64
        """

        if generator.output_shape != discriminator.input_shape:
            raise ValueError('The generator output and discriminator input does not have the same shape!')
        if len(generator.input_shape) != 2:
            raise ValueError('Generator must have 2D shape (batch_size, input_size)')

        self.generator = generator
        self.discriminator = discriminator

        # Get the input shapes. The first dimension is the batch_size so we throw it away.
        self.d_input_shape = tuple(discriminator.input_shape[1:])
        self.g_input_dim = generator.input_shape[1]  # generator should only have 1 dimension

        self.d_input = Input(shape=self.d_input_shape)
        self.g_input = Input(shape=(self.g_input_dim,))

        # Generator setup
        self.d_output_for_g_train = _wrap_layer(discriminator, trainable=False)(
            _wrap_layer(generator)(self.g_input)
        )

        # Discriminator setup
        self.d_output_on_real = _wrap_layer(discriminator)(self.d_input)
        self.g_output_for_d_train = _wrap_layer(generator, trainable=False)(self.g_input)
        self.d_output_on_fake = _wrap_layer(discriminator)(self.g_output_for_d_train)

    def compile(
        self,
        g_optimizer,
        d_optimizer,
        is_wgan=False,
        wgan_gradient_lambda=0.0,
    ):
        """Compile the GAN model to make it training ready.

        Args:
            g_optimizer (keras Optimizer or str): Optimizer for the GAN generator.
            d_optimizer (keras Optimizer or str): Optimizer for the GAN discriminator.
            is_wgan (bool): Whether to use the WGAN loss function or the original Jenson-Shannon GAN loss.
            wgan_gradient_lambda (float): For WGAN mode only; the weight of the gradient penalty in improved WGAN.
        """
        if wgan_gradient_lambda < 0:
            raise ValueError('wgan_gradient_lambda must be >= 0')

        # Generator trainer
        if is_wgan:
            g_transform = lambda x: -x
        else:
            # The model expects output of 2D shape (batch_size, 1), so we expand the dimension here
            g_transform = lambda x: loss_func.gan_generator_loss(x)[:, None]
        g_loss_each = layers.Lambda(g_transform)(self.d_output_for_g_train)
        self.g_trainer_model = Model(inputs=self.g_input, outputs=g_loss_each)
        self.g_trainer_model.compile(g_optimizer, loss=loss_func.identity_loss)

        # Discriminator trainer
        if is_wgan:
            d_loss_each = layers.add([layers.Lambda(lambda x: -x)(self.d_output_on_real), self.d_output_on_fake])
            if wgan_gradient_lambda:
                penalty = _wgan_gradient_penalty(
                    self.d_input,
                    self.g_output_for_d_train,
                    _wrap_layer(self.discriminator),
                )
                d_loss_each = layers.add([
                    d_loss_each,
                    layers.Lambda(lambda x, p=wgan_gradient_lambda: x * p)(penalty),
                ])
        else:
            # The model expects output of 2D shape (batch_size, 1), so we expand the dimension here
            d_loss_each = layers.Lambda(
                lambda x: loss_func.gan_discriminator_loss(x[0], x[1])[:, None]
            )([self.d_output_on_real, self.d_output_on_fake])
        self.d_trainer_model = Model(inputs=[self.d_input, self.g_input], outputs=d_loss_each)
        self.d_trainer_model.compile(d_optimizer, loss=loss_func.identity_loss)

    def generator_train_on_batch(self, batch_size=64):
        """Train the generator with a mini-batch.

        Args:
            batch_size (int): The size of mini-batch for generator training

        Returns:
            The scalar generator loss.
        """
        random_g_input = np.random.normal(size=(batch_size, self.g_input_dim))
        return self.g_trainer_model.train_on_batch(random_g_input, np.empty(batch_size))

    def discriminator_train_on_batch(self, x, sample_weight=None):
        """Train the discriminator with a mini-batch.

        Args:
            x (ndarray): Input mini-batch of training data.
            sample_weight (ndarray): Sample weights of the input.

        Returns:
            The scalar discriminator loss.
        """
        batch_size = x.shape[0]
        random_g_input = np.random.normal(size=(batch_size, self.g_input_dim))
        return self.d_trainer_model.train_on_batch(
            [x, random_g_input],
            np.empty(batch_size),
            sample_weight=sample_weight,
        )


def _wrap_layer(layer, trainable=True):
    wrapped = Sequential([layer])
    wrapped.trainable = trainable
    return wrapped


def _wgan_gradient_penalty(x_real, x_gen, discriminator):
    def interpolate(input_args):
        real, gen = input_args
        batch_size = K.shape(real)[0]
        epsilon_shape = [batch_size] + [1] * (K.ndim(real) - 1)
        epsilon = K.random_uniform(epsilon_shape)
        return epsilon * real + (1 - epsilon) * gen

    def grad_penalty(input_args):
        output, x_inte = input_args
        batch_size = K.shape(x_inte)[0]
        grad = K.gradients(output, x_inte)
        grad_reshape = K.reshape(grad, (batch_size, -1))
        grad_norm = K.sqrt(K.sum(K.square(grad_reshape), axis=1))
        # The model expects output of 2D shape (batch_size, 1), so we expand the dimension here
        return K.square(grad_norm - 1)[:, None]

    x_interpolate = layers.Lambda(interpolate)([x_real, x_gen])
    d_output = discriminator(x_interpolate)
    return layers.Lambda(grad_penalty, output_shape=(1,))([d_output, x_interpolate])

