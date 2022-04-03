from sklearn import datasets
import numpy as np

import matplotlib.pyplot as plt

#Загрузка датасета
digits = datasets.load_digits()


#Показать случайные картинки
print(digits.data.shape)
print(digits.target)
fig, axes = plt.subplots(4,4)
axes=axes.flatten()
for i, ax in enumerate(axes):
    dig_ind=np.random.randint(0,len(digits.images))
    ax.imshow(digits.images[dig_ind].reshape(8,8))
    ax.set_title(digits.target[dig_ind])
plt.show()


#Посчитать картинок какого класса сколько
dic={x:0 for x in range(10)}
for dig in digits.target:
    dic[dig]+=1
print(dic)

def prepare_data(data, avg):
    """
    Подготавливает данные для кореляционного классификатора
    :param data: np.array, данные (размер выборки, количество пикселей
    :param avg: float, параметр для предобработки
    :return: data: np.array, данные (размер выборки, количество пикселей
    """
    pass

def train_val_test_split(data, labels):
    """
    Делит выборку на обучающий, валидационный и тестовый датасеты
    :param data: np.array, данные (размер выборки, количество пикселей)
    :param labels: np.array, метки (размер выборки,)
    :return: train_data, train_labels, validation_data, validation_labels, test_data, test_labels
    """
    pass

def softmax(vec):
    pass

class CorrelationClassifier:

    def __init__(self, classes_count=10):
        self.classes_count=classes_count

    def fit(self, data, labels):
        """
        Производит обучение алгоритма на заданном датасете
        :param data: np.array, данные (размер выборки, количество пикселей)
        :param labels: np.array, метки (размер выборки,)
        :return:
        """
        self.averages=[]
        pass
        self.averages=np.array(self.averages)

    def predict(self, data):
        """
        Предсказывает вектор вероятностей для каждого наблюдения в выборке
        :param data: np.array, данные (размер выборки, количество пикселей)
        :return: np.array, результаты (len(data), count_of_classes)
        """
        pass


    def accuracy(self, data, labels):
        """
        Оценивает точность (accuracy) алгоритма по выборке
        :param data: np.array, данные (размер выборки, количество пикселей)
        :param labels: np.array, метки (размер выборки,)
        :return:
        """
        pass


train_data, train_labels, validation_data, validation_labels, test_data, test_labels = train_val_test_split(digits.data, digits.target)

train_data = prepare_data(train_data, 8)
validation_data = prepare_data(validation_data, 8)
test_data = prepare_data(test_data, 8)

#Посчитать картинок какого класса сколько в обучающем датасете
dic={x:0 for x in range(10)}
for dig in train_labels:
    dic[dig]+=1
print(dic)


classifier=CorelationClassifier()
classifier.fit(train_data, train_labels)
print(f"Training accuracy {classifier.accuracy(train_data, train_labels)}")
print(f"Validation accuracy {classifier.accuracy(validation_data, validation_labels)}")



