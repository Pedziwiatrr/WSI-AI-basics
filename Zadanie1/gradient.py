import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Nr indeksu: 331421 => A = 1, B = 2, C = 4
# 1: f(x) = x + 2sin(x),    D = (-4pi, 4pi)
# 2: g(x, y) = 4xy/e^(x^2 + y^2),   Dx = (-2, 2), Dy  (-2, 2)


def f(x):
    return x + 2 * math.sin(x)


def f_derivative(x):
    return (1 + 2 * math.cos(x),)


def g(x, y):
    return 4 * x * y / np.exp(x**2 + y**2)


def g_derivative(x, y):
    return (
        (4 - 8 * x**2) * y * np.exp((-x) ** 2 - (y) ** 2),
        (4 - 8 * y**2) * x * np.exp((-x) ** 2 - (y) ** 2),
    )
    # pochodne cząstkowe (∂x, ∂y)


def grad_descent(
    function, derivative, domain, learning_rate, args, step_count=100, find_min=True
):
    if find_min:
        learning_rate = -learning_rate
        # domyślnie dążymy do minimum. Jeśli find_min = False, funkcja będzie dążyć do maksimum

    path = [args]

    for i in range(step_count):
        new_args = [
            variable + learning_rate * derivative(*args)[i]
            for i, variable in enumerate(args)
        ]
        # args - argumenty funkcji na których obecnie "jesteśmy"
        # każdy z tych argumentów zmieniamy z osobna co każdy krok gradientu przesuwając się po wykresie

        if all(domain[i][0] <= new_args[i] <= domain[i][1] for i in range(len(domain))):
            args = new_args
            path.append(args)
        else:
            break

    if len(args) == 1:
        two_dimensions_chart(function, domain[0], path)
    elif len(args) == 2:
        three_dimensions_chart(function, domain, path)

    return args


def two_dimensions_chart(function, domain, path):
    x_values = [x / 100 for x in range(int(domain[0] * 100), int(domain[1] * 100))]
    y_values = [function(x) for x in x_values]
    plt.plot(
        x_values,
        y_values,
        label=function.__name__ + "(x)",
        linestyle="--",
    )

    path_x = [position[0] for position in path]
    path_y = [function(x) for x in path_x]
    plt.plot(
        path_x,
        path_y,
        color="red",
        linewidth=4,
        label="Ścieżka gradientu",
    )

    plt.scatter(
        path_x[0], f(path_x[0]), color="green", s=75, label="Punkt startowy", zorder=2
    )

    plt.legend()
    plt.savefig(
        "./Zadanie1/wykresy/gradient_wykres_2d.png", dpi=500, bbox_inches="tight"
    )
    # plt.show()


def three_dimensions_chart(function, domain, path):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x_values = np.linspace(domain[0][0], domain[0][1])
    y_values = np.linspace(domain[1][0], domain[1][1])

    X, Y = np.meshgrid(x_values, y_values)
    Z = function(X, Y)
    ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)

    path_x = [position[0] for position in path]
    path_y = [position[1] for position in path]
    path_z = [function(x, y) for x, y in zip(path_x, path_y)]

    ax.plot(path_x, path_y, path_z, color="red", linewidth=5, label="Ścieżka gradientu")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    random_x = random.uniform(-4 * math.pi, 4 * math.pi)
    args = grad_descent(
        f, f_derivative, [(-4 * math.pi, 4 * math.pi)], 0.01, [random_x], 1000
    )
    print(args)

    random_x = random.uniform(-2, 2)
    random_y = random.uniform(-2, 2)
    args = grad_descent(g, g_derivative, [(-2, 2), (-2, 2)], 0.01, [random_x, random_y])
    print(args)
