import numpy as np
import sys

funcs = {
    'sin(x)': lambda x: np.sin(x),
    'cos(x)': lambda x: np.cos(x),
    'ln(x)': lambda x: special_log(x),
    'e^x': lambda x: np.exp(x),
    'sqrt(x)': lambda x: np.sqrt(x),
    'x': lambda x: x,
    'x^2': lambda x: x ** 2,
    'x^3': lambda x: x ** 3,
    '1': lambda x: x ** 0
}


def special_log(x):
    try:
        ln = np.log(x)
    except:
        "((((((((((((((("
    if np.shape(ln) == ():  # Если x - число
        if ln == -np.Inf:
            ln = -sys.maxsize
    else:  # Если x - вектор
        ln[ln == -np.Inf] = -sys.maxsize

    return ln


def create_design_matrix_polynomial_regression(x, polynom_degree):
    N = x.size  # Количество элементов в выборке
    Fi = []
    for i in range(polynom_degree):
        Fi.append(x ** i)
    Fi = np.reshape(Fi, (polynom_degree, N)).T
    return Fi


def create_design_matrix(x, function_names):
    N = x.size  # Количество элементов в выборке
    Fi = []
    for name in function_names:
        Fi.append(funcs[name](x))
    Fi = np.reshape(Fi, (len(function_names), N)).T
    return Fi


def eval_w(Fi, t):
    return np.linalg.inv(Fi.T @ Fi) @ Fi.T @ t


def eval_y_poly(x, w):
    fi_x = [x ** i for i in range(0, len(w))]
    return w.T @ fi_x


def eval_y(x, w, function_names):
    fi_x = [funcs[name](x) for name in function_names]
    return w.T @ fi_x


def eval_MSE_poly(x, t, w):
    N = t.size
    s = sum([(t[i] - w.T @ eval_fi_xn_poly(x[i], len(w))) ** 2 for i in range(1, N)])
    return s / N


def eval_MSE(t, w, Fi):
    N = t.size
    s2 = sum([(t[i] - w.T @ Fi[i]) ** 2 for i in range(0, N)])
    return s2 / N


def eval_fi_xn_poly(xn, len):
    return [xn ** i for i in range(0, len)]


def eval_fi_xn(xn, f_names):
    return [funcs[f_name](xn) for f_name in f_names]
