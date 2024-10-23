import math
import random

# Nr indeksu: 331421 => A = 1, B = 2, C = 4
# 1: f(x) = x + 2sin(x),    D = (-4pi, 4pi)
# 2: g(x, y) = 4xy/e^(x^2 + y^2),   Dx = (-2, 2), Dy  (-2, 2)


def f(x):
    if -4 * math.pi < x < 4 * math.pi:
        return x + 2 * math.sin(x)
    else:
        return None


def g(x, y):
    if -2 < x < 2 and -2 < y < 2:
        return 4 * x * y / math.e ** (x**2 + y**2)
    else:
        return None


def grad_descent(function, learning_rate, position, step_count=100, find_min=True):
    if find_min:
        learning_rate = -learning_rate
    for i in range(step_count):
        new_position = position + learning_rate  # * pochodna
        position = new_position
