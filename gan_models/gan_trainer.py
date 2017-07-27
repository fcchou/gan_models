"""
Model class to help training GAN.
"""
from typing import List, Union

import numpy as np
import keras
from keras.models import Model, Layer, Input
from keras.optimizers import Optimizer
import keras.backend as K

from gan_models.common import Tensor
from gan_models import model_layers, loss_func


class GanTrainer(Model):

    def __init__(
        self,
        generator: Layer,
        discriminator: Layer,
        name: str='gan_trainer',
    ):
        """Keras Model object for training GAN model.

        Args:
            generator:
                A keras model for GAN generator.
                The input should be a 1D noise vector of Uniform(-1, 1).
                The output shape should match the input shape of discriminator
            discriminator:
                A keras model for GAN discriminator.
                The discriminator should return the single raw value of the last layer (before e.g. sigmoid).
            noise_generator:
                A callable that takes an int dimension, and outputs a keras layer
                generating random noise in the dimension, for generator input.
            name: Name of the model.

        Examples:
            See examples/dcgan_mnist.py for how to use this class.
        """

        if generator.output_shape != discriminator.input_shape:
            raise ValueError('The generator output and discriminator input does not have the same shape!')
        if len(generator.input_shape) != 2:
            raise ValueError('Generator must have 2D shape (batch_size, input_size)')

        self.generator = generator
        self.discriminator = discriminator

        # Get the input shapes. The first dimension is the batch_size so we throw it away.
        d_input_shape = tuple(discriminator.input_shape[1:])
        g_input_dim = generator.input_shape[1]  # generator should only have 1 dimension

        self.d_input = Input(shape=d_input_shape)
        self.g_input = model_layers.NormalNoiseGenerator(g_input_dim)(self.d_input)

        self.g_output = generator(self.g_input)
        self.d_output_on_input = discriminator(self.d_input)
        self.d_output_on_generated = discriminator(self.g_output)

        super().__init__(
            inputs=self.d_input,
            outputs=[self.d_output_on_generated, self.d_output_on_input],
            name=name,
        )

    def gan_compile(
        self,
        g_optimizer: Union[Optimizer, str],
        d_optimizer: Union[Optimizer, str],
        is_wgan: bool=False,
        wgan_gradient_lambda: float=0,
        **kwargs
    ) -> None:
        """Compile the GAN model to make training ready.

        Args:
            optimizer: A keras optimizer object, or a string (e.g. 'adam').
            **kwargs: Additional kwargs for self.compile
        """
        # Compile with a placeholder loss; we don't actually use it.
        self.compile(g_optimizer, 'mean_squared_error', **kwargs)
        self.g_optimizer = keras.optimizers.get(g_optimizer)
        self.d_optimizer = keras.optimizers.get(d_optimizer)
        if is_wgan:
            self.g_loss = -K.mean(self.d_output_on_generated)
            self.d_loss = -K.mean(self.d_output_on_input) + K.mean(self.d_output_on_generated)
            if wgan_gradient_lambda:
                self.d_loss += wgan_gradient_lambda * self.wgan_gradient_penalty
        else:
            self.g_loss = loss_func.gan_generator_loss(self.d_output_on_generated)
            self.d_loss = loss_func.gan_discriminator_loss(self.d_output_on_input, self.d_output_on_generated)
        # Hack the loss and metrics, ignore the dummy loss set in compile
        self.total_loss = self.d_loss
        self.metrics_tensors = [self.g_loss]

    @property
    def wgan_gradient_penalty(self):
        """Gradient penalty as detailed in the improved WGAN paper (https://arxiv.org/abs/1704.00028)."""
        batch_size = K.shape(self.d_input)[0]
        epsilon_shape = [batch_size] + [1] * (K.ndim(self.d_input) - 1)
        epsilon = K.random_uniform(epsilon_shape)
        x_hat = epsilon * self.d_input + (1 - epsilon) * self.g_output
        d_loss_x_hat = self.discriminator(x_hat)
        grad = K.gradients(d_loss_x_hat, x_hat)
        grad_reshape = K.reshape(grad, (batch_size, -1))
        grad_norm = K.sqrt(K.sum(K.square(grad_reshape), axis=1))
        return K.mean(K.square(grad_norm - 1))

    @property
    def g_updates(self):
        """Update ops for generator training."""
        return self.g_optimizer.get_updates(self.generator.trainable_weights, self.constraints, self.g_loss)

    @property
    def d_updates(self):
        """Update ops for discriminator training."""
        return self.d_optimizer.get_updates(self.discriminator.trainable_weights, self.constraints, self.d_loss)

    def _get_train_function(self, train_updates: List[Tensor], losses: List[Tensor], name: str):
        inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
        if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]

        return K.function(
            inputs,
            losses,
            updates=self.updates + train_updates,
            name=name,
            **self._function_kwargs
        )

    def _make_train_function(self):
        if not hasattr(self, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')
        if self.train_function is None:
            self.train_function = self._get_train_function(
                train_updates=self.d_updates + self.g_updates,
                losses=[self.d_loss, self.g_loss],
                name='train_function',
            )
            self.g_train_func = self._get_train_function(
                train_updates=self.g_updates,
                losses=[self.g_loss],
                name='g_train_function',
            )
            self.d_train_func = self._get_train_function(
                train_updates=self.d_updates,
                losses=[self.d_loss],
                name='d_train_function',
            )

    def gan_fit(self, x: np.ndarray, **kwargs):
        """Fit GAN model.

        This function does a fast variant that updates both discriminator and generator in one go,
        by reusing the generator pass in discriminator training for updating generator.
        The idea comes from https://ctmakro.github.io/site/on_learning/fast_gan_in_keras.html

        Note that this forces you to train one mini-batch on discriminator and one mini-batch on generator
        in alternation; to train discriminator and generator separately, use the generator_train_on_batch and
        discriminator_train_on_batch function.

        Args:
            x: Input training data.
            **kwargs: Additional kwarg as in self.fit

        Returns:
            A keras training History object.
        """
        return self.fit(x=x, y=_get_dummy_labels(x), **kwargs)

    def gan_train_on_batch(self, x: np.ndarray, sample_weight: np.ndarray=None):
        """Train GAN with a mini-batch of data. This uses the alternating fast training described in self.gan_fit.

        Args:
            x: Input mini-batch of training data.
            sample_weight: Weights of the input, see self.train_on_batch.

        Returns:
            The list of losses [d_loss, g_loss].
        """
        return self.train_on_batch(x=x, y=_get_dummy_labels(x), sample_weight=sample_weight)

    def generator_train_on_batch(self, x: np.ndarray):
        """Train the generator with a mini-batch.

        Args:
            x: Input mini-batch of training data.
                Note that the generator does not actually need any input to train; the only thing being used
                is the batch size of the input. So any arbitrary input that match the shape of discriminator input
                can be used.

        Returns:
            The scalar generator loss.
        """
        # Hack self.train_function so we can reuse the train_on_batch code
        self._make_train_function()
        orig_train_func = self.train_function
        try:
            self.train_function = self.g_train_func
            return self.gan_train_on_batch(x)
        finally:
            self.train_function = orig_train_func

    def discriminator_train_on_batch(self, x: np.ndarray, sample_weight: np.ndarray=None):
        """Train the discriminator with a mini-batch.

        Args:
            x: Input mini-batch of training data.

        Returns:
            The scalar discriminator loss.
        """
        # Hack self.train_function so we can reuse the train_on_batch code
        self._make_train_function()
        orig_train_func = self.train_function
        try:
            self.train_function = self.d_train_func
            return self.gan_train_on_batch(x, sample_weight=sample_weight)
        finally:
            self.train_function = orig_train_func


def _get_dummy_labels(input_x, output_dim=2):
    input_size = input_x.shape[0]
    dummy_label = np.empty(input_size)
    return [dummy_label] * output_dim
