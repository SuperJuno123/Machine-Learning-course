from sklearn import datasets
from sklearn import model_selection
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

digits = datasets.load_digits()
eps0 = 0.1


def data_split(data, labels, n, train_size=0.6, valid_size=0.5):
    train_set, v_set, train_labels, v_labels = d_split(data, labels, train_size)
    validation_set, test_set, validation_labels, test_labels = d_split(v_set, v_labels, valid_size)
    while not_balanced(train_labels, n):
        train_set, v_set, train_labels, v_labels = d_split(data, labels, train_size)
    while not_balanced(validation_labels, n):
        validation_set, test_set, validation_labels, test_labels = d_split(v_set, v_labels, valid_size)
    return train_set, train_labels, validation_set, validation_labels, test_set, test_labels


def d_split(data, labels, train_size):
    return model_selection.train_test_split(data, labels, train_size=train_size)


def not_balanced(labels, n):
    vect_classes = np.zeros(n)
    for digit in labels:
        vect_classes[digit] += 1
    avg = len(labels) / n
    for each_class in vect_classes:
        if -eps0 * avg > (each_class - avg) / avg > eps0 * avg:
            return True
    return False


def normalize_data(data):
    _min = np.amin(data, axis=0)
    _max = np.amax(data, axis=0)
    deltas = _max - _min
    for delta in deltas:
        if delta == 0:
            delta = 1
    return (data - _min) / delta


def to_OHE_vector(labels, n, m):
    v = np.zeros((n, m))
    for i in range(n):
        v[i][labels[i]] = 1
    return v


def from_OHE_vector(one_hot_encoding_vect):
    return np.argmax(one_hot_encoding_vect, axis=1)


class SoftmaxRegression:
    def __init__(self, data, labels):
        self.classes_count = 10
        self.train_data, \
        self.train_labels, \
        self.v_data, \
        self.v_labels, \
        self.test_data, \
        self.test_labels = data_split(data, labels, self.classes_count)
        self.train_data = normalize_data(self.train_data)
        self.w = np.ones((self.classes_count, self.train_data.shape[1]))
        self.b = np.ones(self.classes_count)
        self.OHEv = to_OHE_vector(self.train_labels, len(self.train_labels), self.classes_count)

    def z(self, w, b, x):
        return w @ x + b

    def stable_softmax(self, x_vect):
        exps = np.exp(x_vect - np.amax(x_vect))
        return exps / np.sum(exps)

    def gradient_w_b(self, num):
        y = self.stable_softmax(self.z(self.w, self.b, self.train_data[num]))
        t = self.OHEv[num]
        return y - t

    def gradient_w(self, num, gamma=0.01):
        y = self.stable_softmax(self.z(self.w, self.b, self.train_data[num]))
        t = self.OHEv[num]
        y_t = y - t
        w_v = np.zeros((len(y_t), len(self.train_data[num])))
        for i in range(len(y_t)):
            for j in range(len(self.train_data[num])):
                w_v[i][j] = y_t[i] * self.train_data[num][j]
        w_v -= gamma * self.w
        return (w_v)

    def gradient_descent(self, n=1000, m=100, alpha=0.2, batch_size=10, is_draw=True, is_print=True, is_pickle=False):
        losses = []
        accuracies = []
        for i in range(0, n):
            rand = random.randint(0, len(self.train_data) - batch_size)
            average_grad_w = np.zeros((len(self.w), len(self.train_data[0])))
            average_grab_b = np.zeros(len(self.w))
            average_loss = 0
            for j in range(rand, rand + batch_size):
                average_grad_w += self.gradient_w(j) / batch_size
                average_grab_b += self.gradient_w_b(j) / batch_size
                average_loss += self.loss(j) / batch_size
            self.w -= alpha * average_grad_w
            self.b -= alpha * average_grab_b
            if is_draw:
                if i % m == 0:
                    accuracies.append(self.accuracy(self.confusion_matrix(self.v_data, self.v_labels)))
                losses.append(average_loss)
            if is_pickle:
                if i % m == 0:
                    self.PickleSave(self.w, self.b, 'pickle-' + str(i))

        if is_draw:
            self.draw(losses, accuracies, n, m)

        if is_print:
            self.print_metrics()

    def draw(self, losses, accuracies, n, m):
        fig, ax = plt.subplots()
        plt.title('Функция потерь')
        plt.xlabel('Число итераций n')
        plt.ylabel('f')
        ax.plot(np.linspace(1, n, n), losses)

        fig2, ax2 = plt.subplots()
        plt.title('Точность на валидационной выборке')
        plt.xlabel('Число итераций n')
        plt.ylabel('accuracy')
        ax2.plot(np.linspace(0, n - n % m, n // m), accuracies)
        plt.show()

    def print_metrics(self):
        conf_matrix = self.confusion_matrix(self.test_data, self.test_labels)
        print(f"Confusion matrix:\n{conf_matrix}\n"
              f"Precision: {self.precision(conf_matrix)}\n"
              f"Recall: {self.recall(conf_matrix)}]\n"
              f"Accuracy: {self.accuracy(conf_matrix)}")

    def loss(self, num):
        z = self.z(self.w, self.b, self.train_data[num])
        k = -np.amax(z)
        y = self.stable_softmax(z)
        return np.sum(-y * (z + k - np.log(np.sum(np.exp(z + k)))))

    def confusion_matrix(self, data, labels):
        conf_matrix = np.zeros((self.classes_count, self.classes_count))
        for i in range(len(data)):
            conf_matrix[np.argmax(self.stable_softmax(self.z(self.w, self.b, data[i])))][labels[i]] += 1
        return conf_matrix

    def precision(self, conf_matrix):
        t = np.zeros(len(conf_matrix))
        for i in range(len(conf_matrix)):
            t[i] = conf_matrix[i][i]
        return t / np.sum(conf_matrix, axis=1)

    def recall(self, conf_matrix):
        t = np.zeros(len(conf_matrix))
        for i in range(len(conf_matrix)):
            t[i] = conf_matrix[i][i]
        return t / np.sum(conf_matrix, axis=0)

    def accuracy(self, conf_matrix):
        return self.TP_and_TN(conf_matrix) / np.sum(conf_matrix)

    def TP_and_TN(self, conf_matrix):
        TP_and_TN = 0
        for i in range(len(conf_matrix)):
            TP_and_TN += conf_matrix[i][i]
        return TP_and_TN

    def PickleSave(self, w, b, file):
        data = {'w': w, 'b': b}
        with open(file, 'wb') as f:
            pickle.dump(data, f)
        f.close()

    def PickleLoad(self, w, b, file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            w = data['w']
            b = data['b']
            f.close()


softmax_regression = SoftmaxRegression(digits.data, digits.target)
softmax_regression.gradient_descent()
