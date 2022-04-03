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

# Создать 3 выборки: обучающую, валидационную, тестовую (80:10:10) предварительно перемешав данные.
# Данные взять с 1 задания
def generate_tree_samples(x, t):
    total_amount = len(x)
    indexes=np.arange(total_amount)
    np.random.shuffle(indexes)
    x=x[indexes]
    t=t[indexes]

    training_sample_size = int(total_amount * 0.8)
    validation_sample_size = int(total_amount * 0.1)
    test_sample_size = int(total_amount * 0.1)

    return (x[:training_sample_size], t[:training_sample_size]), \
           (x[training_sample_size: training_sample_size + validation_sample_size],
            t[training_sample_size: training_sample_size + validation_sample_size]), \
           (x[training_sample_size + validation_sample_size:], t[training_sample_size + validation_sample_size:])


