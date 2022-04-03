import numpy as np


# а) Рост футболистов(класс 0) - numpy.random.randn(500) * 20 + 160
#
# б) Рост баскетболистов(класс 1) - numpy.random.randn(500) * 10 + 190

def random(height, crutch):
    return np.random.randint(2)

def dummy_height(height, compare_height):
    if height>compare_height:
        return 1
    else:
        return 0