import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('x_train 데이터의 형태 : ', x_train.shape)
print('x_train[0] 데이터의 형태 : ', x_train[0].shape)
print('y_train 데이터의 형태 : ', y_train.shape)

num = x_train[0]
for i in range(28):
  for j in range(28):
    print('{:4d}'.format(num[i][j]), end='')
  print()

plt.imshow(num, cmap='Greys', interpolation='nearest')
print('y_train[0] = ', y_train[0])