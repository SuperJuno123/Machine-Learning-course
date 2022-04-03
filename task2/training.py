import evaluation

def training(training_set, validation_set):
    x_train, t_train = training_set
    x_val, t_val = validation_set

    from itertools import combinations

    dict_fun_for_comb = evaluation.funcs.copy()
    del dict_fun_for_comb['1']
    names = list(combinations(dict_fun_for_comb, 1))
    names.extend(combinations(dict_fun_for_comb, 2))
    names.extend(combinations(dict_fun_for_comb, 3))

    all_names = []

    MSE_all_val = []
    MSE_all_tr = []
    w_all = []

    for name in names:
        function_names = [f for f in name]
        function_names.insert(0, '1')  # фи0=1
        all_names.append(function_names)

        Fi_x_train = evaluation.create_design_matrix(x_train, function_names)
        weights = evaluation.eval_w(Fi_x_train, t_train)
        w_all.append(weights)

        tr_MSE = evaluation.eval_MSE(t_train, weights, Fi_x_train)
        MSE_all_tr.append(tr_MSE)

        Fi_x_val = evaluation.create_design_matrix(x_val, function_names)
        validation_MSE = evaluation.eval_MSE(t_val, weights, Fi_x_val)
        MSE_all_val.append(validation_MSE)

    return all_names, w_all, MSE_all_tr, MSE_all_val