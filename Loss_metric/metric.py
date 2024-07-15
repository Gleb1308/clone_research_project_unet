import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import numpy as np
import pandas as pd

class Dice_score(tf.keras.metrics.Metric):

  def __init__(self, name='Dice_score', **kwargs):
    super(Dice_score, self).__init__(name=name, **kwargs)
    self.dice_sum = self.add_weight(name='dice_sum', initializer='zeros')
    self.count = self.add_weight(name='count', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    self.count.assign_add(1.0)
    y_true = tf.cast(K.flatten(y_true), self.dtype)
    y_pred = tf.cast(K.flatten(y_pred)>0, self.dtype)
    intersection = K.sum(y_true * y_pred)
    dice = (2*intersection +1.0) / (K.sum(y_true) + K.sum(y_pred) + 1.0)
    dice = tf.cast(dice, self.dtype)
    self.dice_sum.assign_add(dice)

  def result(self):
    return self.dice_sum/self.count
