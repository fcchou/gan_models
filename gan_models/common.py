try:
    import tensorflow as tf
    Tensor = tf.Tensor
except ImportError:
    import typing
    Tensor = object