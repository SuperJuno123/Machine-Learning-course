import numpy as np
import generating_data

# Negative = 0 = футболист
# Positive = 1 = баскетболист

footballer = generating_data.generate_heights(500, 0)
basketballer = generating_data.generate_heights(500, 1)

import metrics
import classificators

FP, TN, TP, FN, accuracy, precision, recall = metrics.eval_metrics(classificators.random, class_0=footballer,
                                                                   class_1=basketballer)

print("Для случайного классификатора FP =", FP, "TN =", TN, "TP =", TP, "FN =", FN, "accuracy =", accuracy,
      "precision =", precision, "recall =", recall)

FP, TN, TP, FN, accuracy, precision, recall = metrics.eval_metrics(classificators.dummy_height, class_0=footballer,
                                                                   class_1=basketballer)

print("Для ростового классификатора FP =", FP, "TN =", TN, "TP =", TP, "FN =", FN, "accuracy =", accuracy,
      "precision =", precision, "recall =", recall)

coordinates = []
for height in range(90, 230, 10):
    _, _, _, _, _, precision, recall = metrics.eval_metrics(classificators.dummy_height, class_0=footballer,
                                                            class_1=basketballer, fixed_height=height)
    coordinates.append((recall, precision))

import matplotlib.pyplot as plt

plt.style.use('seaborn-pastel')

# coordinates.sort(key=lambda x: x[0])
x = [x[0] for x in coordinates]
y = [x[1] for x in coordinates]

plt.plot(x, y)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True, color='gray', linestyle=':')

from sklearn.metrics import auc
print("Площадь под графиком, вычисленная методом трапеций: ", auc(x, y))

plt.savefig("Precision-Recall Curve")
plt.show()

