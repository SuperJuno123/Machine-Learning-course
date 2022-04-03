import evaluation
import numpy as np
import generating_data
import training
import warnings
warnings.filterwarnings("ignore")


x_input, t_input = generating_data.generate_x_t()

training_set, validation_set, test_set = generating_data.generate_tree_samples(x_input, t_input)

x_train, t_train = training_set
x_val, t_val = validation_set

all_names, w_all, MSE_all_tr, MSE_all_val = training.training(training_set, validation_set)

import sys

min_mse = [sys.maxsize for i in range(3)]
min_i = [sys.maxsize for i in range(3)]
for i in range(len(MSE_all_val)):
    if (MSE_all_val[i] < min_mse).any():
        index = np.argmax(min_mse)
        min_mse[index] = MSE_all_val[i]
        min_i[index] = i

x_test, t_test = test_set
index_of_bestest = min_i[np.argmin(min_mse)]
Fi_test = evaluation.create_design_matrix(x_test, all_names[index_of_bestest])
w_test = evaluation.eval_w(Fi_test, t_test)
MSE_test = evaluation.eval_MSE(t_test, w_test, Fi_test)

print("Для лучшей модели " + "".join(
    [f"{w:.2f}*{name}+" for w, name in zip(w_all[index_of_bestest][1:], all_names[index_of_bestest][1:])])
      + f"{w_test[0]:.2f} "
      + "точность на тестовой выборке составляет " + str(MSE_test))

name_of_bests = [all_names[index] for index in min_i]
weights_of_bests = [w_all[index] for index in min_i]
MSE_tr_of_bests = [MSE_all_tr[index] for index in min_i]
MSE_val_of_bests = [MSE_all_val[index] for index in min_i]

import visualization

visualization.draw_hysto(name_of_bests, weights_of_bests, MSE_tr_of_bests, MSE_val_of_bests, save_pic=True)
