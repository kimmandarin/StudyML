import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255, x_test / 255

model = keras.models.Sequential([
                                 keras.layers.Flatten(input_shape = (28, 28)),
                                 keras.layers.Dense(256, activation = 'relu'),
                                 keras.layers.Dense(10, activation = 'softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(tf.expand_dims(x_train, axis=-1), y_train, epochs=5)

model.summary()

print("신경망 모델의 학습 결과 : ")
eval_loss, eval_acc = model.evaluate(x_test, y_test)
print("test 데이터의 손실값", eval_loss, "test 데이터의 정확도", eval_acc)