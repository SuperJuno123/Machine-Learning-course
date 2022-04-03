import classificators
import numpy as np


def eval_metrics(classificator, class_0, class_1, fixed_height=165):
    N = len(class_0) + len(class_1)
    if N == 0:
        raise ValueError('У тебя данных нет! Ты как из палаты сбежал, уважаемый?')
        return

    result_0 = np.array([classificator(class_0[i], fixed_height) for i in range(len(class_0))])

    FP = np.count_nonzero(result_0)  # сколько раз сказал, что принадлежит классу 1 (на самом деле принадлежит классу 0)
    TN = len(result_0) - FP # сколько раз сказал, что принадлежит классу 0 (на самом деле принадлежит классу 0)

    result_1 = np.array([classificator(class_1[i], fixed_height) for i in range(len(class_1))])

    TP = np.count_nonzero(result_1) # сколько раз сказал, что принадлежит классу 1 (на самом деле принадлежит классу 1)
    FN = len(result_1) - TP # сколько раз сказал, что принадлежит классу 0 (на самом деле принадлежит классу 1)

    accuracy = (TP + TN) / N

    if (TP + FP) == 0:
        precision = 1
    else:
        precision = TP / (TP + FP)

    if (TP + FN) == 0:
        recall = 1
    else:
        recall = TP / (TP + FN)

    return FP, TN, TP, FN, accuracy, precision, recall
