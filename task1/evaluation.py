import numpy as np


def create_design_matrix(x, polynom_degree):
    N = x.size         # Количество элементов в выборке
    Fi = []
    for i in range(polynom_degree):
        Fi.append(x ** i)
    Fi = np.reshape(Fi, (polynom_degree, N)).T
    return Fi


def eval_w(Fi, t):
    return np.linalg.inv(Fi.T @ Fi) @ Fi.T @ t

def eval_y(x, w):
    fi_x = [x ** i for i in range(0, len(w))]
    return w.T @ fi_x

def eval_MSE(x, t, w):
    N = t.size
    s = sum([(t[i] - w.T @ eval_fi_xn(x[i], len(w))) ** 2 for i in range(1, N)])
    return s / N

def eval_fi_xn(xn, len):
    return [xn ** i for i in range(0, len)]
