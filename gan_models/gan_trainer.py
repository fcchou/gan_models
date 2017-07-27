"""
Model class to help training GAN.
"""
from typing import Callable, List, Union

import numpy as np
from keras.models import Model, Layer, Input
from keras.optimizers import Optimizer
import keras.backend as K

from gan_models.common import Tensor
from gan_models import model_layers


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

        d_input = Input(shape=d_input_shape)
        g_input = model_layers.UniformNoiseGenerator(g_input_dim)(d_input)

        g_output = generator(g_input)
        self.d_output_on_input = discriminator(d_input)
        self.d_output_on_generated = discriminator(g_output)

        super().__init__(
            inputs=d_input,
            outputs=[self.d_output_on_generated, self.d_output_on_input],
            name=name,
        )

    def gan_compile(
        self,
        optimizer: Union[Optimizer, str],
        generator_loss_func: Callable[[Tensor], Tensor],
        discriminator_loss_func: Callable[[Tensor, Tensor], Tensor],
        **kwargs
    ) -> None:
        """Compile the GAN model to make training ready.

        Args:
            optimizer: A keras optimizer object, or a string (e.g. 'adam').
            generator_loss_func:
                A callable computing the generator loss.
                The input is the single tensor from D(G(z)) (discriminator applied to generator output),
                and it outputs a scalar loss value.
            discriminator_loss_func:
                A callable computing the discriminator loss, taking 2 arguments.
                The first argument is D(x) (discriminator on real input x), the second argument is D(G(z)).
                Outputs a scalar loss value.
            **kwargs: Additional kwargs for self.compile
        """
        # Compile with a placeholder loss; we don't actually use it.
        self.compile(optimizer, 'mean_squared_error', **kwargs)
        self.g_loss = generator_loss_func(self.d_output_on_generated)
        self.d_loss = discriminator_loss_func(self.d_output_on_input, self.d_output_on_generated)
        # Hack the loss and metrics, ignore the dummy loss set in compile
        self.total_loss = self.d_loss
        self.metrics_tensors = [self.g_loss] + self.metrics_tensors

    @property
    def g_updates(self):
        """Update ops for generator training."""
        return self.optimizer.get_updates(self.generator.trainable_weights, self.constraints, self.g_loss)

    @property
    def d_updates(self):
        """Update ops for discriminator training."""
        return self.optimizer.get_updates(self.discriminator.trainable_weights, self.constraints, self.d_loss)

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
                losses=[self.total_loss] + self.metrics_tensors,
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
