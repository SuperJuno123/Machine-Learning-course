# Код для генерации данных

import numpy as np

x = np.linspace(0, 2 * np.pi, 1000)

y = 100 * np.sin(x) + 0.5 * np.exp(x) + 300

error = 10 * np.random.randn(1000)

t = y + error

f = open("source.txt", 'w')

for i in range(x.size):
    f.write(str(x[i]) + ' ' + str(t[i]) + '\n')


def generate_x_t():
    x = np.linspace(0, 2 * np.pi, 1000)

    y = 100 * np.sin(x) + 0.5 * np.exp(x) + 300

    error = 10 * np.random.randn(1000)

    t = y + error

    return x, t