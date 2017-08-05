"""
Useful layers and models for creating GAN models.
"""
from collections import namedtuple

from keras.models import Sequential
from keras.constraints import Constraint
from keras import layers
import keras.backend as K


# Parameters for convolutional layers
ConvParams = namedtuple('ConvParams', ('filters', 'kernel_sizes', 'strides'))


class UniformNoiseGenerator(layers.Layer):

    def __init__(self, output_dim, val_range=1.0, **kwargs):
        """Generate uniform noise for the data generator.

        The output is random uniform numbers of shape (batch_size, output_dim).
        The input vectors are discarded

        Args:
            output_dim (int): Dimension of the output.
            val_range (float): Range of the uniform noise to be generated ([-val_range, val_range]).
        """
        self.val_range = val_range
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        batch_size = K.shape(inputs)[0]
        return K.random_uniform((batch_size, self.output_dim), minval=-self.val_range, maxval=self.val_range)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class NormalNoiseGenerator(layers.Layer):

    def __init__(self, output_dim, stdev=1.0, **kwargs):
        """Generate uniform noise for the data generator.

        The output is random uniform numbers of shape (batch_size, output_dim).
        The input vectors are discarded

        Args:
            output_dim (int): Dimension of the output.
            stdev (float): Standard deviation of the Gaussian noise.
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
        """Clip weight constraint. The layer weights are constrained to be in [-clip_value, clip_value]

        Args:
            clip_value (float): Size of the weight clipping.
        """
        self.clip_value = clip_value

    def __call__(self, w):
        return K.clip(w, -self.clip_value, self.clip_value)

    def get_config(self):
        return {'clip_value': self.clip_value}


def get_dcgan_generator(input_dim, shape_layer1, conv_params_list, use_batch_norm=True, name='dcgan_generator'):
    """Create DCGAN generator.

    Args:
        input_dim (int): Dimension of the input (random) vector for generation
        shape_layer1 (Tuple[int, int, int]): Shape of the 1st layer before Conv2DTranspose (3 dimensions).
        conv_params_list (List[ConvParams]): List of parameters for the Conv2DTranspose layers.
        use_batch_norm (bool): If batch normalization is used between layers.
        name (str): Name of the model.

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
    input_shape,
    conv_params_list,
    use_batch_norm=True,
    clip_weight=None,
    name='dcgan_discriminator',
):
    """DCGAN discriminator.

    Args:
        input_dim (int): Dimension of the input (random) vector for generation
        shape_layer1 (Tuple[int, int, int]): Shape of the 1st layer before Conv2DTranspose (3 dimensions).
        conv_params_list (List[ConvParams]): List of parameters for the Conv2D layers.
        use_batch_norm (bool): If batch normalization is used between layers.
        clip_weight (bool): If clip-weight constraint is applied to the weights.
        name (str): Name of the model.

    Returns:
        A Keras Sequential model as DCGAN discriminator.
    """
    if clip_weight is not None and clip_weight <= 0:
        raise ValueError('clip_weight must be > 0')

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
