import numpy as np
import matplotlib.pyplot as plt

import evaluation

np.random.seed(665)

lambda_reg=0.000
gamma=0.007

lambda_all = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 5, 10, 20, 30, 40, 50, 60, 70]

points=100
poly_deg=15
x_set=np.linspace(0,1,points)
gt_set=30*x_set*x_set
err=2*np.random.randn(points)
err[3]+=200
err[77]+=100
err[50]-=100

special_error = [200, 100, -100]
x_with_special_err = [x_set[3], x_set[77], x_set[50]]
t_with_special_err = [30*x**2+error for x, error in zip(x_with_special_err,special_error)]

t_set=gt_set+err

t_set = np.delete(t_set, [3, 77, 50])
x_set = np.delete(x_set, [3, 77, 50])

import generating_data
training_set, validation_set, test_set = generating_data.generate_tree_samples(x_set,t_set,
                                                                               training_sample_size=80-3,
                                                                               validation_sample_size=10,
                                                                               test_sample_size=10)

x_train, t_train = training_set

new_places = np.random.randint(0,len(x_train), 3)
x_train = np.insert(x_train, new_places, x_with_special_err)
t_train = np.insert(t_train, new_places, t_with_special_err)

x_val, t_val = validation_set

info_for_graphic = []

min_val_MSE = 9999999
lamb_with_min = 2342526

lambs=[]
MSE_train=[]
MSE_val=[]

for lamb in lambda_all:
    loss_vals, w_itog=evaluation.gradient_descent(x_train, t_train, poly_deg, gamma, lamb)
    MSE_train = evaluation.eval_MSE_poly(x_train, t_train, w_itog.T, poly_deg)
    MSE_val = evaluation.eval_MSE_poly(x_val, t_val, w_itog.T, poly_deg)
    if MSE_val < min_val_MSE:
        min_val_MSE = MSE_val
        lamb_with_min = lamb
        w_with_min = w_itog

    import visualization
    visualization.draw_grad_method(x_train, t_train, loss_vals, w_itog, poly_deg)

    # print("MSE val: ", MSE_val)
    info_for_graphic.extend((lamb, MSE_train, MSE_val))

info_for_graphic = np.reshape(info_for_graphic, (len(lambda_all),3))

x_test, t_test = test_set
MSE_test = evaluation.eval_MSE_poly(x_test, t_test, w_with_min.T, poly_deg)
print("Для модели с лучшей точностью на валидационной выборке, равной", min_val_MSE, "с коэффициентом регуляризации", lamb_with_min,
      "оценкой на тестовой выборке является", MSE_test)


import visualization

# visualization.draw_grad_method(x_train, t_train, loss_vals, w_itog, poly_deg)
visualization.draw_hysto_all_lambda(info_for_graphic, save_pic=True)