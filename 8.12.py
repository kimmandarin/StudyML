import numpy as np

def cee(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def mse(y, t):
    return ((y-t)**2).mean()

t = np.array([0, 0, 1])
y_hat1 = np.array([0.4, 0.05, 0.55])
y_hat2 = np.array([0.9, 0.09, 0.01])

print("y_hat1과의 mse : {:.2f}".format(cee(y_hat1, t)))
print("y_hat2과의 mse : {:.2f}".format(cee(y_hat2, t)))
print("두 값의 비 : {:.2f}".format(cee(y_hat2, t)/cee(y_hat1, t)))

print("y_hat1과의 mse : {:.2f}".format(mse(y_hat1, t)))
print("y_hat2과의 mse : {:.2f}".format(mse(y_hat2, t)))
print("두 값의 비 : {:.2f}".format(mse(y_hat2, t)/mse(y_hat1, t)))