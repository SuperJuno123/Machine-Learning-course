import numpy as np


# 1) Сгенерируйте выборку двух классов по 500 элементов в каждой:
#
#    а) Рост футболистов (класс 0) - numpy.random.randn(500) * 20 + 160
#
#    б) Рост баскетболистов (класс 1) - numpy.random.randn(500) * 10 + 190

def generate_heights(n, _class):
    if _class == 0:  # футболисты
        return np.random.randn(n) * 20 + 160
    if _class == 1:
        return np.random.randn(n) * 10 + 190
