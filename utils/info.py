import platform
import numpy as np
import tensorflow as tf
from tensorflow.python.util import object_identity


def trainable_params(model):
    """
    Counts the number of trainable parameters in a Keras model.
    """
    count_params = lambda weights: int(
        sum(
            np.prod(p.shape.as_list())
            for p in object_identity.ObjectIdentitySet(weights)
        )
    )
    if hasattr(model, "_collected_trainable_weights"):
        return count_params(model._collected_trainable_weights)
    else:
        return count_params(model.trainable_weights)


def hardware():
    """
    Detects platform and accelerator.
    """
    if platform.system() == "Darwin":
        if platform.processor() == "arm":
            from tensorflow.python.compiler.mlcompute import mlcompute

            mlcompute.set_mlc_device(device_name="gpu")
            return "Apple M1"
        else:
            return "Apple Intel"
    elif len(tf.config.list_physical_devices("GPU")) > 0:
        return "NVIDIA"
    return "Linux Intel"