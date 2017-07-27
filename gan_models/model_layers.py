"""
Useful layers and models for creating GAN models.
"""
from collections import namedtuple
from typing import Tuple, Sequence

from keras.models import Sequential
from keras.constraints import Constraint
from keras import layers
import keras.backend as K


# Parameters for convolutional layers
ConvParams = namedtuple('ConvParams', ('filters', 'kernel_sizes', 'strides'))


class UniformNoiseGenerator(layers.Layer):

    def __init__(self, output_dim: int, minval: float=-1.0, maxval: float=1.0, **kwargs):
        """Generate uniform noise for the data generator.

        The output is random uniform numbers of shape (batch_size, output_dim).
        The input vectors are discarded

        Args:
            output_dim: Dimension of the output.
            minval: Min value of the uniform noise.
            maxval: Max value of the uniform noise.
        """
        self.minval = minval
        self.maxval = maxval
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        batch_size = K.shape(inputs)[0]
        return K.random_uniform((batch_size, self.output_dim), minval=self.minval, maxval=self.maxval)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class NormalNoiseGenerator(layers.Layer):

    def __init__(self, output_dim: int, stdev: float=1.0, **kwargs):
        """Generate uniform noise for the data generator.

        The output is random uniform numbers of shape (batch_size, output_dim).
        The input vectors are discarded

        Args:
            output_dim: Dimension of the output.
            stdev: Standard deviation of the Gaussian noise.
        """
        self.stdev = stdev
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        batch_size = K.shape(inputs)[0]
        return K.random_normal((batch_size, self.output_dim), stddev=self.stdev)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

class ClipWeight(Constraint):

    def __init__(self, clip_value=0.01):
        self.clip_value = clip_value

    def __call__(self, w):
        return K.clip(w, -self.clip_value, self.clip_value)

    def get_config(self):
        return {'clip_value': self.clip_value}


def get_dcgan_generator(
    input_dim: int,
    shape_layer1: Tuple[int, int, int],
    conv_params_list: Sequence[ConvParams],
    use_batch_norm: bool=True,
    name: str='dcgan_generator',
) -> Sequential:
    """Create DCGAN generator.

    Args:
        input_dim: Dimension of the input (random) vector for generation
        shape_layer1: Shape of the 1st layer before Conv2DTranspose (3 dimensions).
        conv_params_list: List of parameters for the Conv2DTranspose layers.
        name: Name of the model.

    Returns:
        A Keras Sequential model as DCGAN generator.
    """
    model_layers = [
        layers.Dense(input_dim=input_dim, units=(shape_layer1[0] * shape_layer1[1] * shape_layer1[2])),
    ]
    if use_batch_norm:
        model_layers.append(layers.BatchNormalization())
    model_layers += [
        layers.Activation('relu'),
        layers.Reshape(shape_layer1),
    ]
    for conv_params in conv_params_list[:-1]:
        model_layers.append(
            layers.Conv2DTranspose(
                filters=conv_params.filters,
                kernel_size=conv_params.kernel_sizes,
                strides=conv_params.strides,
                padding='same',
            )
        )
        if use_batch_norm:
            model_layers.append(layers.BatchNormalization())
        model_layers.append(layers.Activation('relu'))
    # For the output layer, don't use batch-norm, and use sigmoid activation so the output is 0-1.
    conv_params = conv_params_list[-1]
    model_layers.append(
        layers.Conv2DTranspose(
            filters=conv_params.filters,
            kernel_size=conv_params.kernel_sizes,
            strides=conv_params.strides,
            padding='same',
            activation='sigmoid',
        )
    )
    return Sequential(layers=model_layers, name=name)


def get_dcgan_discriminator(
    input_shape: Tuple[int, int, int],
    conv_params_list: Sequence[ConvParams],
    use_batch_norm: bool=True,
    clip_weight: float=None,
    name: str='dcgan_discriminator',
) -> Sequential:
    """DCGAN discriminator.

    Args:
        input_shape: Shape of the input tensor (3 dimensions).
        conv_params_list: List of parameters for the Conv2D layers.
        name: Name of the model.

    Returns:
        A Keras Sequential model as DCGAN discriminator.
    """
    assert clip_weight is None or clip_weight > 0
    # First layer, which does not use batch-norm
    constraint = None if clip_weight is None else ClipWeight(clip_weight)
    conv_params = conv_params_list[0]
    model_layers = [
        layers.Conv2D(
            input_shape=input_shape,
            filters=conv_params.filters,
            kernel_size=conv_params.kernel_sizes,
            strides=conv_params.strides,
            padding='same',
            kernel_constraint=constraint,
            bias_constraint=constraint,
        ),
        layers.LeakyReLU(0.2),
    ]

    for conv_params in conv_params_list[1:]:
        model_layers.append(
            layers.Conv2D(
                input_shape=input_shape,
                filters=conv_params.filters,
                kernel_size=conv_params.kernel_sizes,
                strides=conv_params.strides,
                padding='same',
                kernel_constraint=constraint,
                bias_constraint=constraint,
            )
        )
        if use_batch_norm:
            model_layers.append(layers.BatchNormalization())
        model_layers.append(layers.LeakyReLU(0.2))
    model_layers += [
        layers.Flatten(),
        layers.Dense(1, kernel_constraint=constraint, bias_constraint=constraint),
    ]
    return Sequential(layers=model_layers, name=name)

