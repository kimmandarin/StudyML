import  numpy as np

def softmax(a):
    exp_of_a = np.exp(a)
    sum_exp = np.sum(exp_of_a)
    y = exp_of_a / sum_exp
    return y

a = np.array([0.5, 4.1, 2.5, 5.6, 1.2]) * 2
print("신경망의 예측값 : ", a)
print("소프트맥스 함수의 출력 : ", softmax(a))
print("소프트맥스 함수의 최대값 : ", np.max(softmax(a)))