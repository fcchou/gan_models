"""
Model class to help training GAN.

The main class GanTrainer hacks some keras Model class internals to perform faster training and to reuse the
mini-batching and progress bar of model.fit function.

See Also
"""
import contextlib

import numpy as np
import keras
from keras.models import Model, Input
import keras.backend as K

from gan_models import model_layers, loss_func


class GanTrainer(Model):

    def __init__(self, generator, discriminator, name='gan_trainer'):
        """Keras Model object for training GAN model.

        Args:
            generator (keras Model):
                A keras model for GAN generator.
                The input should be a 1D noise vector of Uniform(-1, 1).
                The output shape should match the input shape of discriminator
            discriminator (keras Model):
                A keras model for GAN discriminator.
                The discriminator should return the single raw value of the last layer (before e.g. sigmoid).
            name (str): Name of the model.

        Examples:
            >>> model = GanTrainer(generator=generator, discriminator=discriminator)
            >>> model.gan_compile(g_optimizer='rmsprop', d_optimizer='rmsprop')  # Initialize training with optimizers
            >>> model.gan_fit(x_train)  # Train the model
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
        g_optimizer,
        d_optimizer,
        n_discriminator_training=1,
        is_wgan=False,
        wgan_gradient_lambda=0.0,
        **kwargs
    ) -> None:
        """Compile the GAN model to make it training ready.

        Args:
            g_optimizer (keras Optimizer or str): Optimizer for the GAN generator.
            d_optimizer (keras Optimizer or str): Optimizer for the GAN discriminator.
            n_discriminator_training (int): Number of steps for discriminator update for each generator update.
            is_wgan (bool): Whether to use the WGAN loss function or the original Jenson-Shannon GAN loss.
            wgan_gradient_lambda (float): For WGAN mode only; the weight of the gradient penalty in improved WGAN.
            **kwargs: Additional kwargs for self.compile
        """
        if n_discriminator_training < 1:
            raise ValueError('n_discriminator_training must be >= 1')
        if wgan_gradient_lambda < 0:
            raise ValueError('wgan_gradient_lambda must be >= 0')

        self.g_optimizer = keras.optimizers.get(g_optimizer)
        self.d_optimizer = keras.optimizers.get(d_optimizer)
        self._n_discriminator_training = n_discriminator_training

        # Compile with a placeholder loss; we don't actually use it.
        self.compile(g_optimizer, 'mean_squared_error', **kwargs)

        sample_weight = self.sample_weights[0]
        if is_wgan:
            # Use WGAN loss function as described in the paper
            self.g_loss = -K.mean(self.d_output_on_generated * sample_weight)
            d_loss_col = self.d_output_on_generated - self.d_output_on_input
            if wgan_gradient_lambda:
                d_loss_col += (
                    wgan_gradient_lambda
                    * loss_func.get_wgan_gradient_penalty(self.d_input, self.g_output, self.discriminator)
                )
            self.d_loss = K.mean(d_loss_col * sample_weight)
        else:
            # Standard Jensen-Shannon loss for GAN
            self.g_loss = K.mean(loss_func.gan_generator_loss(self.d_output_on_generated) * sample_weight)
            self.d_loss = K.mean(
                loss_func.gan_discriminator_loss(self.d_output_on_input, self.d_output_on_generated)
                * sample_weight
            )
        # Hack the loss and metrics, ignore the dummy loss set in compile
        self.total_loss = self.d_loss
        self.metrics_tensors = [self.g_loss]

    @property
    def g_updates(self):
        """Update ops for generator training."""
        return self.g_optimizer.get_updates(self.generator.trainable_weights, self.constraints, self.g_loss)

    @property
    def d_updates(self):
        """Update ops for discriminator training."""
        return self.d_optimizer.get_updates(self.discriminator.trainable_weights, self.constraints, self.d_loss)

    def _get_train_function(self, train_updates, losses, name):
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
        if not hasattr(self, '_train_function'):
            raise RuntimeError('You must compile your model before using it.')
        if not hasattr(self, 'gd_train_function'):
            # A fast train function variant that updates D and G together in one session run.
            self.gd_train_function = self._get_train_function(
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
                losses=[self.d_loss, self.g_loss],
                name='d_train_function',
            )

    @property
    def train_function(self):
        """Default GAN train_function.

        If n_discriminator_training is larger than 1, it performs a hack that
        updates D+G every n_discriminator_training rounds; in other rounds only D is updated.
        """
        def f(*args, **kwargs):
            f.counter += 1
            if f.counter == f.n_discriminator:
                f.counter = 0
                return self.gd_train_function(*args, **kwargs)
            else:
                return self.d_train_func(*args, **kwargs)

        if self._train_function is not None:
            # returns the hacked train function; see self._hack_train_function
            return self._train_function
        elif self._n_discriminator_training == 1:
            return self.gd_train_function
        else:
            f.counter = 0
            f.n_discriminator = self._n_discriminator_training
            return f

    @train_function.setter
    def train_function(self, value):
        """Dummy set function to maintain compatibility with the Model class"""
        self._train_function = None
        pass

    @contextlib.contextmanager
    def _hack_train_function(self, new_func):
        """Context that hacks the return of self.train_function."""
        try:
            self._train_function = new_func
            yield
        finally:
            self._train_function = None

    def gan_fit(self, x, **kwargs):
        """Fit GAN model.

        This function does a fast variant that updates both discriminator and generator in one go,
        by reusing the generator pass in discriminator training for updating generator.
        The idea comes from https://ctmakro.github.io/site/on_learning/fast_gan_in_keras.html

        The fast training trick is most efficient when you update discriminator and
        generator 1:1 alternatively (i.e. n_discriminator_training=1), which makes it ~30% faster.

        Args:
            x (ndarray): Input training data.
            **kwargs: Additional kwarg as in self.fit

        Returns:
            A keras training History object.
        """
        return self.fit(x=x, y=self._get_dummy_labels(x), **kwargs)

    def gan_train_on_batch(self, x, sample_weight=None):
        """Train GAN with a mini-batch of data. This uses the alternating fast training described in self.gan_fit.

        Args:
            x (ndarray): Input mini-batch of training data.
            sample_weight (ndarray): Sample weights of the input, see self.train_on_batch.

        Returns:
            The list of losses [d_loss, g_loss].
        """
        return self.train_on_batch(x=x, y=self._get_dummy_labels(x), sample_weight=sample_weight)

    def generator_train_on_batch(self, x):
        """Train the generator with a mini-batch.

        Args:
            x (ndarray): Input mini-batch of training data.
                Note that the generator does not actually need any input to train; the only thing being used
                is the batch size of the input. So any arbitrary input that match the shape of discriminator input
                can be used.

        Returns:
            The scalar generator loss.
        """
        self._make_train_function()
        # Hack self.train_function so we can reuse the train_on_batch code
        with self._hack_train_function(self.g_train_func):
            return self.gan_train_on_batch(x)

    def discriminator_train_on_batch(self, x, sample_weight=None):
        """Train the discriminator with a mini-batch.

        Args:
            x (ndarray): Input mini-batch of training data.
            sample_weight (ndarray): Sample weights of the input, see self.train_on_batch.

        Returns:
            The list of losses [d_loss, g_loss]. Note that g_loss is returned although not optimized against.
        """
        self._make_train_function()
        # Hack self.train_function so we can reuse the train_on_batch code
        with self._hack_train_function(self.d_train_func):
            return self.gan_train_on_batch(x, sample_weight=sample_weight)

    @staticmethod
    def _get_dummy_labels(input_x, output_dim=2):
        return [np.empty(input_x.shape[0])] * output_dim
