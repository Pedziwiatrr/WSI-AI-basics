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


def f_derivative(x):
    if -4 * math.pi < x < 4 * math.pi:
        return 1 + 2 * math.cos(x)
    else:
        return None


def g(x, y):
    if -2 < x < 2 and -2 < y < 2:
        return 4 * x * y / math.e ** (x**2 + y**2)
    else:
        return None


def g_derivative(x, y):
    if -2 < x < 2 and -2 < y < 2:
        return (
            (4 - 8 * x**2) * y * math.e ** ((-x) ** 2 - (y) ** 2),
            (4 - 8 * y**2) * x * math.e ** ((-x) ** 2 - (y) ** 2),
        )
        # pochodne cząstkowe (∂x, ∂y)
    else:
        return None


def grad_descent(derivative, learning_rate, position, step_count=100, find_min=True):
    if find_min:
        learning_rate = -learning_rate
        # domyślnie dążymy do minimum. Jeśli find_min = False, funkcja będzie dążyć do maksimum
    for i in range(step_count):
        new_position = [
            variable + learning_rate * derivative(*position) for variable in position
        ]
        position = new_position

    return position


if __name__ == "__main__":
    position = grad_descent(f_derivative, 0.01, [f(random.uniform(-2, 2))], 1000)
    print(position)
