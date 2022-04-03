

def read(file):
    f = open(file)
    lines = f.readlines()
    x = []
    t = []
    for line in lines:
        line = line.split(" ")
        x.append(float(line[0]))
        t.append(float(line[1]))
    import numpy as np
    return np.array(x), np.array(t)