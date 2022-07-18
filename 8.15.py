import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255, x_test / 255

model = keras.models.Sequential([
                                 keras.layers.Flatten(input_shape = (28, 28)),
                                 keras.layers.Dense(64, activation = 'relu'),
                                 keras.layers.Dense(64, activation = 'relu'),
                                 keras.layers.Dense(64, activation = 'relu'),
                                 keras.layers.Dense(64, activation = 'relu'),
                                 keras.layers.Dense(64, activation = 'relu'),
                                 keras.layers.Dense(10, activation = 'softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(tf.expand_dims(x_train, axis=-1), y_train, epochs = 10, validation_data = (x_test, y_test))
pd.DataFrame(history.history).plot(figsize = (10, 6))