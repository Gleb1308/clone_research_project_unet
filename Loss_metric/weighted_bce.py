import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

def weighted_binary_crossentropy(target, output, weights):
    weights = tf.convert_to_tensor(weights, dtype=target.dtype)

    #epsilon_ = tf.constant(tf.keras.backend.epsilon(), output.dtype.base_dtype)
    epsilon_ = 1e-19
    output = tf.math.abs(tf.clip_by_value(output, epsilon_, 1.0 - epsilon_))

    # Compute cross entropy from probabilities.
    bce = weights[1] * target * tf.math.log(output)
    bce += weights[0] * (1 - target) * tf.math.log(1 - output)
    return -bce

class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, weights = [0.85, 0.15], name="weighted_binary_crossentropy", fn = None):
        super().__init__()
        self.weights = weights
        self.name = name
        self.fn = weighted_binary_crossentropy if fn is None else fn

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.math.sigmoid(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        return tf.reduce_mean(self.fn(y_true, y_pred, self.weights))
