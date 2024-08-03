# Метод релаксации

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

values = np.array([-18, -18, 8, 132, 176])  # Значения всех аргументов
startX = np.array([15, 15])  # Начальная точка
e = 0.1

def func(x, values): # вычисляет значение функции
    return values[0] * x[0] ** 2 + values[1] * x[1] ** 2 + values[2] * x[0] * x[1] + values[3] * x[0] + values[4] * x[1]

def derivative(x, values): # вычисляет значение производной
    dfX = np.array([2 * values[0] * x[0] + values[2] * x[1] + values[3], 2 * values[1] * x[1] + values[2] * x[0] + values[4]])
    return dfX

def hesse(values): # вычисляет матрицу Гессе
    return np.array([[values[0] * 2, values[2]], [values[2], values[1] * 2]])

def plot_graph(x_1, x_2, scale, x, y, text): # строит траекторию
    plt.contour(x_1, x_2, scale, levels=50, alpha=.5)
    plt.plot(x, y, label=text, color='black')
    plt.legend()
    plt.show()

def print_table(steps, x1, x2, f, text): # строит таблицу
    table = PrettyTable()
    table.add_column("Step", steps)
    table.add_column("x1", x1)
    table.add_column("x2", x2)
    table.add_column("f(X)", f)
    print(text)
    print(table)

def relax(startX, values, e):
    x1scale = np.arange(4, 16, 0.1)
    x2scale = np.arange(5, 16, 0.1)
    x1scale, x2scale = np.meshgrid(x1scale, x2scale)
    scale = func([x1scale, x2scale], values)
    X = startX
    H = hesse(values)
    fX = func(X, values)
    dfX = derivative(X, values)
    i = 0
    steps = [0]
    x1 = [X[0]]
    x2 = [X[1]]
    f = [fX]
    K = np.array([0, dfX[1]])
    t = -(dfX.dot(K)) / (K.dot(H).dot(K))

    while np.linalg.norm(dfX) > e:
        X = X + t * K
        fX = func(X, values)
        dfX = derivative(X, values)
        f.append(fX)
        x1.append(X[0])
        x2.append(X[1])
        K = dfX
        t = -(dfX.dot(K)) / (K.dot(H).dot(K))
        i += 1
        steps.append(i)

    plot_graph(x1scale, x2scale, scale, x1, x2, 'Метод релаксации')
    print_table(steps, x1, x2, f, "Метод релаксации:")

relax(startX, values, e)