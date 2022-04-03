# Код с примером постройки любого количества графиков в одном окне:

import matplotlib.pyplot as plt
import numpy as np

# fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
# axes[0].plot([5, 5, 5, 5])
# # axes[0].set_title("прямая")
# axes[1].plot([0, 1, 4, 9])
# axes[1].set_title("квадратичная")
# axes[2].plot([0, 1, 8, 29])
# axes[2].set_title("кубическая")
# plt.show()


# fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
#
# axes[0].plot([i ** 2 for i in range(10)])

# plt.show()

def draw_graphics(num_of_graphics, x, t, y, save_pic=False):

    fig, axes = plt.subplots(ncols=5, nrows=4, sharex=True, sharey=True, figsize=(10,6))
    k = 0

    for i in range(4):
        for j in range(5):
            axes[i, j].plot(x, t, marker=',', linewidth=0, color = "#66CDAA")
            axes[i, j].set_title(str(k + 1) + " степень", fontweight='semibold', fontstyle='italic', fontsize=6, color='#000000')
            if k < num_of_graphics:
                axes[i, j].plot(x, y[k], color = 'red')
            k+=1

    plt.axis([0, 2 * np.pi, 200, 600])
    if save_pic:
        plt.savefig("Graphics")
    plt.show()

def draw_MSE(mse, k,  save_pic=False):
    plt.plot([i + 1 for i in range(0,k)], mse, label="Среднее всех ошибок", marker=".")
    plt.legend()
    if save_pic:
        plt.savefig("MSE")
    plt.show()