from sklearn import datasets
import numpy as np

import matplotlib.pyplot as plt

# Загрузка датасета
digits = datasets.load_digits()

# Посчитать картинок какого класса сколько
dic = {x: 0 for x in range(10)}
for dig in digits.target:
    dic[dig] += 1
print(dic)


def prepare_data(data, avg):
    """
    Подготавливает данные для кореляционного классификатора (вычтем из каждой картинки теоретическое среднее)
    :param data: np.array, данные (размер выборки, количество пикселей
    :param avg: float, параметр для предобработки
    :return: data: np.array, данные (размер выборки, количество пикселей
    """
    return data - avg


def train_val_test_split(data, labels):
    """
    Делит выборку на обучающий, валидационный и тестовый датасеты
    :param data: np.array, данные (размер выборки, количество пикселей)
    :param labels: np.array, метки (размер выборки,)
    :return: train_data, train_labels, validation_data, validation_labels, test_data, test_labels
    """
    N = len(data)
    n_train_data = int(np.round(0.8 * N))
    n_valid_data = int(np.round(0.1 * N))
    # n_test_data = int(np.round(0.1 * N))

    indexes = np.arange(len(digits.data))
    np.random.shuffle(indexes)

    train_data = data[indexes[:n_train_data]]
    train_labels = labels[indexes[:n_train_data]]

    validation_data = data[indexes[n_train_data:n_train_data + n_valid_data]]
    validation_labels = labels[indexes[n_train_data:n_train_data + n_valid_data]]

    test_data = data[indexes[n_train_data + n_valid_data:]]
    test_labels = labels[indexes[n_train_data + n_valid_data:]]

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def softmax(vec):
    D = -max(vec)
    vec = np.array(vec) + D
    exps = np.exp(vec)
    return exps / np.sum(exps)


class CorrelationClassifier:

    def __init__(self, classes_count=10):
        self.classes_count = classes_count

    def fit(self, data, labels):
        """
        Производит обучение алгоритма на заданном датасете
        :param data: np.array, данные (размер выборки, количество пикселей)
        :param labels: np.array, метки (размер выборки,)
        :return:
        """
        self.averages = []  # Среднее (эталоны)
        for digit_name in range(10):
            avg_for_cur_digit = np.zeros(64)
            count_cur_digit = 0
            for i in range(labels.shape[0]):
                if labels[i] == digit_name:
                    avg_for_cur_digit += data[i]
                    count_cur_digit += 1
            avg_for_cur_digit = avg_for_cur_digit / count_cur_digit
            self.averages.append(avg_for_cur_digit)
        self.averages = np.array(self.averages)

    def predict(self, data):
        """
        Предсказывает вектор вероятностей для каждого наблюдения в выборке
        :param data: np.array, данные (размер выборки, количество пикселей)
        :return: np.array, результаты (len(data), count_of_classes)
        """
        predictions = []
        for pic in data:
            predict = [sum(average * pic) for average in self.averages]
            predict=softmax(predict)
            predictions.append(predict)
        predictions = np.array(predictions)
        return predictions

    def accuracy(self, data, labels):
        """
        Оценивает точность (accuracy) алгоритма по выборке
        :param data: np.array, данные (размер выборки, количество пикселей)
        :param labels: np.array, метки (размер выборки,)
        :return:
        """
        predictions = self.predict(data)

        matches = 0

        for label, prediction in zip(labels, predictions):
            if np.argmax(prediction) == label:
                matches += 1

        return matches / predictions.shape[0]



def my_max(vect):
    index = np.argmax(vect)
    vect = np.zeros(len(vect))
    vect[index] = index
    return vect


train_data, train_labels, validation_data, validation_labels, test_data, test_labels = train_val_test_split(digits.data,
                                                                                                            digits.target)

train_data = prepare_data(train_data, 8)
validation_data = prepare_data(validation_data, 8)
test_data = prepare_data(test_data, 8)

# Посчитать картинок какого класса сколько в обучающем датасете
print("\nВ обучающем датасете:")
dic = {x: 0 for x in range(10)}
for dig in train_labels:
    dic[dig] += 1
print(dic)

classifier = CorrelationClassifier()
classifier.fit(train_data, train_labels)
print(f"\nTraining accuracy {classifier.accuracy(train_data, train_labels)}")
print(f"Validation accuracy {classifier.accuracy(validation_data, validation_labels)}")

from sklearn.metrics import confusion_matrix
y_true = test_labels
predictions = classifier.predict(test_data)
y_pred = [np.argmax(prediction) for prediction in predictions]
conf_matrix = confusion_matrix(y_true, y_pred, labels=[i for i in range(10)])


# Посчитать картинок какого класса сколько в обучающем датасете
print("\nВ тестовом датасете:")
dic = {x: 0 for x in range(10)}
for dig in test_labels:
    dic[dig] += 1
print(dic)

print(conf_matrix)

for i in range(10):
    print(f"Precision for class {i}: { conf_matrix[i,i] / sum(conf_matrix[i])}")
    print(f"Recall for class {i}: { conf_matrix[i,i] / sum(conf_matrix[:,i])}")


fig, axes = plt.subplots(2, 5)
for i in range(2):
    for j in range(5):
        axes[i, j].xaxis.set_major_locator(plt.NullLocator())
        axes[i, j].yaxis.set_major_locator(plt.NullLocator())

averages = classifier.averages

axes = axes.flatten()
for i, ax in (enumerate(axes)):
    dig_ind = np.random.randint(0, len(digits.images))
    # ax.imshow(digits.images[dig_ind].reshape(8, 8))
    ax.imshow(np.reshape(averages[i], (8, 8)))
    # ax.set_title(digits.target[dig_ind])
    ax.set_title("Эталон цифры " + str(i), fontsize=8)
    
plt.savefig("Averages digit")
plt.show()