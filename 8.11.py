from keras.utils import to_categorical
import numpy as np

data = np.array([0, 1, 2, 3, 4])
print("인코딩할 원본 데이터", data)
encoded = to_categorical(data)
print("원-핫 인코딩된 데이터")
print(encoded)

target = np.array([0, 0, 0, 1, 0])
y_hat = np.array([0.005, 0.173, 0.035, 0.777, 0.01])

def mse(y, t):
    return ((y-t)**2).mean()

print("y_hat과 target과의 오차 : ", mse(y_hat, target))

other_y_hat = np.array([0.2, 0.3, 0.4, 0.01, 0.09])
print("other_y_hat과 target과의 오차 : ", mse(other_y_hat, target))