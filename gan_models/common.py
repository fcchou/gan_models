try:
    import tensorflow as tf
    Tensor = tf.Tensor
except ImportError:
    Tensor = object
