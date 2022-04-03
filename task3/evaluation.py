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


eps=0.001
eps0=0.0001
# lambda_reg=0.000
# gamma=0.007

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


def eval_MSE_poly(x, t, w, poly_deg):
    N = t.size
    Fi = return_phi(x, poly_deg)
    s = sum([(t[i] - w.T @ Fi[i]) ** 2 for i in range(1, N)])
    return float(s / N)


def eval_MSE(t, w, Fi):
    N = t.size
    s2 = sum([(t[i] - w.T @ Fi[i]) ** 2 for i in range(1, N)])
    return float(s2 / N)


def eval_fi_xn_poly(xn, len):
    return [xn ** i for i in range(0, len)]


def eval_fi_xn(xn, f_names):
    return [funcs[f_name](xn) for f_name in f_names]


def return_phi(X, poly_deg):
    phi_n=np.empty((len(X), poly_deg + 1))
    phi_n[:,0]=1
    phi_n[:,1]=X
    for i in range(2, poly_deg + 1):
        phi_n[:,i]=phi_n[:,i-1]*phi_n[:,1]
    return phi_n


def loss(X, t, w, lamb, poly_deg):
    N = len(X)
    Fi = return_phi(X, poly_deg)
    return (1/2) * sum([(t[i] - w @ Fi[i]) ** 2 for i in range(1, N)]) + (lamb / 2) * w @ w.T


# градиент - столбец
def gradient(X, t, w, lamb, poly_deg):
    N = len(X)
    Fi = return_phi(X, poly_deg)
    return -sum([(t[i] - w @ Fi[i]) * Fi[i].T  for i in range(1, N)]) + lamb * w


def gradient_descent(X, t, poly_deg, gamma, lamb):
    loss_vals=[]
    w_next= np.random.rand(poly_deg + 1).reshape((1, poly_deg + 1)) / 100
    w_old = np.zeros(shape=w_next.shape)
    while not almost_equal(w_old, w_next):
        w_old = w_next
        w_next = w_old - gamma * gradient(X, t, w_old, lamb, poly_deg)
        loss_vals.extend(loss(X, t, w_next, lamb, poly_deg))
        # print(loss_vals[-1])
    return loss_vals, w_next


def almost_equal(w_old, w_next):
    return np.linalg.norm(w_next - w_old) < eps * (np.linalg.norm(w_next) + eps0)