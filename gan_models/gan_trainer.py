from typing import Callable, List, Union

import numpy as np
from keras.models import Model, Layer, Input
from keras.optimizers import Optimizer
import keras.backend as K

from gan_models.common import Tensor
from gan_models.model_layers import UniformNoiseGenerator


class GanTrainer(Model):

    def __init__(
        self,
        generator: Layer,
        discriminator: Layer,
        name :str='gan_trainer',
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator

        # Get the input shapes. The first dimension is the batch_size so we throw it away.
        d_input_shape = tuple(discriminator.input_shape[1:])
        g_input_dim = generator.input_shape[1]  # generator should only have 1 dimension

        d_input = Input(shape=d_input_shape)
        g_input = UniformNoiseGenerator(-1.0, 1.0, g_input_dim)(d_input)

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
        # Compile with a placeholder loss; we don't actually use it.
        self.compile(optimizer, 'mean_squared_error', **kwargs)
        self.g_loss = generator_loss_func(self.d_output_on_generated)
        self.d_loss = discriminator_loss_func(self.d_output_on_input, self.d_output_on_generated)
        # Hack the loss and metrics, ignore the dummy loss set in compile
        self.total_loss = self.d_loss
        self.metrics_tensors = [self.g_loss] + self.metrics_tensors

    @property
    def g_updates(self):
        return self.optimizer.get_updates(self.generator.trainable_weights, self.constraints, self.g_loss)

    @property
    def d_updates(self):
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

    def gan_fit(self, x: np.ndarray, **kwargs) -> None:
        self.fit(x=x, y=_get_dummy_labels(x), **kwargs)

    def gan_train_on_batch(self, x: np.ndarray, sample_weight: np.ndarray=None) -> None:
        self.train_on_batch(x=x, y=_get_dummy_labels(x), sample_weight=sample_weight)

    def gan_test_on_batch(self, x: np.ndarray, sample_weight: np.ndarray=None) -> Union[float, List[float]]:
        self.test_on_batch(x=x, y=_get_dummy_labels(x), sample_weight=sample_weight)

    def generator_train_on_batch(self, x: np.ndarray, sample_weight: np.ndarray=None) -> None:
        # Hack self.train_function so we can reuse the train_on_batch code
        self._make_train_function()
        orig_train_func = self.train_function
        try:
            self.train_function = self.g_train_func
            return self.gan_train_on_batch(x, sample_weight=sample_weight)
        finally:
            self.train_function = orig_train_func

    def discriminator_train_on_batch(self, x: np.ndarray, sample_weight: np.ndarray=None) -> None:
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



