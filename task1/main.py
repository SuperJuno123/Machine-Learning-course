import numpy as np
from input_output import read
from generating_data import generate_x_t
import evaluation
import visualization

file_path = 'source.txt'

x_input, t_input = read(file_path)

# Стоит задача поиска оптимальных весов путём минимизации среднеквадратической ошибки

x_gen, t_gen = generate_x_t()

k = 20           # Степень последнего полинома / количество графиков

w_regression_all = []
y_regression_all = []

for i in range(k):
    Fi = evaluation.create_design_matrix(x_gen, i + 1)
    w = evaluation.eval_w(Fi, t_gen)
    y_regression_all.append(evaluation.eval_y(x_gen, w))
    w_regression_all.append(w)
    # print(w)

visualization.draw_graphics(k, x_gen, t_gen, y_regression_all, True)

mean_squared_error = [evaluation.eval_MSE(x_gen,t_gen,w) for w in w_regression_all]

visualization.draw_MSE(mean_squared_error, k, True)