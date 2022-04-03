import matplotlib.pyplot as plt
import numpy as np


plt.style.use('seaborn-pastel')




def draw_hysto_all_lambda(info_for_graphic, save_pic=False):
    labels= info_for_graphic[:,0]
    MSE_train = info_for_graphic[:,1]
    MSE_val = info_for_graphic[:,2]
    width = 0.3
    plt.figure(figsize=(14,6))
    bin_positions = np.array(list(range(len(info_for_graphic))))
    bins_art1 = plt.bar(bin_positions - width / 2, MSE_train, width, label="Точность на обучающей выборке")
    bins_art2 = plt.bar(bin_positions + width / 2, MSE_val, width,
                        label="Точность на валидационной выборке")
    plt.ylabel('Точность')
    plt.xlabel('Значения lambda')
    plt.title('Точность различных архитектур')
    plt.xticks(bin_positions, labels, fontsize=8)
    plt.legend(loc=3)
    for rect in bins_art1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    for rect in bins_art2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    if save_pic:
        plt.savefig("Hystogram-2")
    plt.show()


def draw_grad_method(x, t, loss_vals, w, poly_deg):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    import evaluation
    ax1.plot(x, t, 'ro', markersize=3)
    ax1.plot(x, w.dot(evaluation.return_phi(x, poly_deg).T).flatten(), marker='$А$', linewidth=0, markersize=3,
             color='blue')
    ax2.plot(list(range(1, len(loss_vals) + 1)), loss_vals)
    ax2.set_xlabel("Итерация")
    ax2.set_ylabel("Ошибка")
    plt.show()


def draw_hysto(names, weights, MSE_tr, MSE_val, save_pic=False):
    width = 0.3
    confidences_training = np.array(MSE_tr)
    confidences_validation = np.array(MSE_val)
    func_names = names
    labels = []

    for i in range(3):
        labels.append("".join(
            [f"{w:.2f}*{name}+" + '\n' for w, name in zip(weights[i][1:], func_names[i][1:])])
                      + f"{weights[i][0]:.2f} ")

    bin_positions = np.array(list(range(len(confidences_training))))
    bins_art1 = plt.bar(bin_positions - width/2, confidences_training, width, label="Точность на обучающей выборке")
    bins_art2 = plt.bar(bin_positions + width/2, confidences_validation, width, label="Точность на валидационной выборке")
    plt.ylabel('Точность')
    plt.title('Точность различных архитектур')
    plt.xticks(bin_positions, labels, fontsize=8)
    plt.legend(loc=3)
    for rect in bins_art1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    for rect in bins_art2:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                     f'{height:.2f}',
                     ha='center', va='bottom')

    if save_pic:
        plt.savefig("Hystogram")
    plt.show()



def draw_graphics(num_of_graphics, x, t, y, save_pic=False):
    fig, axes = plt.subplots(1, num_of_graphics, sharex=True, sharey=True, figsize=(10, 6))

    for i in range(num_of_graphics):
        axes[i].plot(x, t, marker=',', linewidth=0, color="#66CDAA")
        axes[i].plot(x, y[i], marker=',', linewidth=0, color='red')

    # plt.axis([0, 2 * np.pi, 200, 600])
    if save_pic:
        plt.savefig("Graphics")
    plt.show()


def draw_MSE(mse, save_pic=False):
    plt.plot([i + 1 for i in range(0, 20)], mse, label="Среднее всех ошибок", marker=".")
    plt.legend()
    if save_pic:
        plt.savefig("MSE")
    plt.show()

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
