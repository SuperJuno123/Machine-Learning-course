# Код как нарисовать красивый график-гистограмму
import matplotlib.pyplot as plt
import numpy as np
width=0.3
confidences=np.array([67.34234234, 92.428423423, 86.52342342])
weights=np.array([3.656759675, 120.23423423,290.1231231231])
func_names=["sin(x)", "e^x"]
reg1="".join([f"{w:.2f}*{name}+" for w, name in zip(weights,func_names)])+f"{weights[-1]:.2f}"
labels=[reg1, "Регрессия2", "Регрессия3"]
bin_positions=np.array(list(range(len(confidences))))
bins_art=plt.bar(bin_positions, confidences, width, label="Точность на обучающей выборке")
plt.ylabel('Точность')
plt.title('Точность различных архитектур')
plt.xticks(bin_positions, labels)
plt.legend(loc=3)
for rect in bins_art:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2., 1.01*height,
            f'{height:.2f}',
            ha='center', va='bottom')
# plt.show()


# Код как перебрать все сочетания из массива
from itertools import combinations
names = combinations([1, 2, 3], 2)
print(list(names))

# Код как правильно помешать два массива одинаковым образом
indexes=np.arange(len(confidences))
print(confidences, weights)
np.random.shuffle(indexes)
print(indexes)
confidences=confidences[indexes]
weights=weights[indexes]
print(confidences, weights)


def sin(x):
    return np.sin(x)


# [sin(x), cos(x), ln(x), e^x, sqrt(x), x, x^2, x^3]
# funcs = {
#     'sin(x)':      lambda x: np.sin(x),
#     'cos(x)':      lambda x: np.cos(x),
#     'ln(x)':       lambda x: np.log(x),
#     'e^x':         lambda x: np.exp(x),
#     'sqrt(x)':     lambda x: np.sqrt(x),
#     'x':           lambda x: x,
#     'x^2':         lambda x: x ** 2,
#     'x^3':         lambda x: x ** 3,
#     '1':           lambda x: x ** 0
# }

import evaluation
from itertools import combinations
dict_fun_for_comb = evaluation.funcs.copy()
del dict_fun_for_comb['1']
names = combinations(dict_fun_for_comb, 2)
x=np.array([i for i in range(1, 20)])
t=np.array([100 * np.sin(x) + 0.5 * np.exp(x) + 300 for x in range(1, 20)])
print(names)
# weights=np.array([3.656759675, 120.23423423, 290.1231231231])

    # 3) Найти 3 лучшие модели (модели с минимальной ошибкой на вал выборке) и отобразить
    # их на графике ошибок с полной расшифровкой получившейся модели (веса+базисные функции)
    # и двумя столбиками: первый - точность на валидационной выборке, второй - точность на обучающей.
# print("".join([str(weight) + "*" + f + "+"
#                for weight, f
#                in zip(np.round(weights, 2), name)]) + str(weights[-1]))
comb_x = []
comb_names = []

MSE_all = []

for name in names: #Беру каждую пару(тройку и т.д.) фукнций
    print(name)
    function_names = [f for f in name]
    # function_names.append('1')
    function_names.insert(0, '1')    #фи0=1
    Fi = evaluation.create_design_matrix(x, function_names)
    weights = evaluation.eval_w(Fi, t)
    current_MSE = evaluation.eval_MSE(x, t, weights, function_names)
    MSE_all.append(current_MSE)
    comb_x.append([evaluation.funcs[fun_name](x) for weight, fun_name in zip(weights, name)])      # сохраняю иксы
    comb_names.append(function_names)         # сохраняю имена функций

# print("purrrr")
# print(comb_x)

i_1 = np.argmin(np.array(MSE_all))

for c, f in zip(comb_x, comb_names):
    print(f, c)
#
# myfun = funcs['e^x']
# print(myfun(x))
#
# print(funcs['e^x'](x))
# print(funcs['x'](x))
# print(funcs['x^2'](x))
import sys
print(np.log(x, ))