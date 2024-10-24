import math
import random
import matplotlib.pyplot as plt
import numpy as np
import time

# autor: Michał Pędziwiatr
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
        (4 - 8 * x**2) * y * np.exp(-(x**2) - (y) ** 2),
        (4 - 8 * y**2) * x * np.exp(-(x**2) - (y) ** 2),
    )
    # pochodne cząstkowe (∂x, ∂y)


def grad_descent(
    function,
    derivative,
    domain,
    learning_rate,
    args,
    step_count=500,
    find_min=True,
    plot=True,
):
    if find_min:
        learning_rate = -learning_rate
        # domyślnie dążymy do minimum (ponieważ algorytm to gradient descend).
        # Jeśli find_min = False, funkcja będzie dążyć do maksimum

    path = [args]

    start_time = time.time()

    for i in range(step_count):
        new_args = [
            variable + learning_rate * derivative(*args)[i]
            for i, variable in enumerate(args)
        ]
        # args - argumenty funkcji na których obecnie "jesteśmy"
        # każdy z tych argumentów zmieniamy z osobna co każdy krok gradientu "przesuwając się" po wykresie

        if all(domain[i][0] <= new_args[i] <= domain[i][1] for i in range(len(domain))):
            args = new_args
            path.append(args)
        else:
            break

    finish_time = time.time()
    total_time = finish_time - start_time

    if plot:
        if len(args) == 1:
            two_dimensions_chart(
                function, domain[0], path, (learning_rate, step_count), total_time
            )
        elif len(args) == 2:
            three_dimensions_chart(
                function, domain, path, (learning_rate, step_count), total_time
            )

    return (args, total_time)


def two_dimensions_chart(function, domain, path, gradient_params, time):
    x_values = np.linspace(domain[0], domain[1])
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
        linewidth=3,
        label="Ścieżka gradientu",
    )

    plt.scatter(
        path_x[0], f(path_x[0]), color="green", s=50, label="Punkt startowy", zorder=2
    )

    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.suptitle(
        f"{function.__name__}(x) / współczynnik długości kroku = {gradient_params[0]}, max ilość kroków = {gradient_params[1]},\n czas = {time:.6f}s",
        fontsize=12,
        color="blue",
    )
    plt.savefig(
        f"./Zadanie1/wykresy/f/f_{gradient_params[0]}_{gradient_params[1]}.png",
        dpi=500,
        bbox_inches="tight",
    )
    # plt.show()


def three_dimensions_chart(function, domain, path, gradient_params, time):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x_values = np.linspace(domain[0][0], domain[0][1])
    y_values = np.linspace(domain[1][0], domain[1][1])

    X, Y = np.meshgrid(x_values, y_values)
    Z = function(X, Y)
    ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r, alpha=0.5)

    path_x = [position[0] for position in path]
    path_y = [position[1] for position in path]
    path_z = [function(x, y) for x, y in zip(path_x, path_y)]

    ax.plot(
        path_x,
        path_y,
        path_z,
        color="red",
        linewidth=3,
        label="Ścieżka gradientu",
    )
    ax.scatter3D(
        path_x[0],
        path_y[0],
        path_z[0],
        color="green",
        s=50,
        label="Punkt startowy",
        zorder=2,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.suptitle(
        f"{function.__name__}(x, y) / współczynnik długości kroku = {gradient_params[0]}, max ilość kroków = {gradient_params[1]},\n czas = {time:.6f}s",
        fontsize=12,
        color="blue",
    )
    plt.savefig(
        f"./Zadanie1/wykresy/g/g_{gradient_params[0]}_{gradient_params[1]}.png",
        dpi=500,
        bbox_inches="tight",
    )
    plt.show()


def generate_test_params():
    learning_rates = [0.1, 0.05, 0.001]
    max_step_counts = [10, 100, 500, 1000, 5000]
    test_params = []

    for lr in learning_rates:
        for msc in max_step_counts:
            test_params.append((lr, msc))

    return test_params


def generate_points(
    random_points=True, domain=[(-4 * math.pi, 4 * math.pi)], min=True, dim=2
):
    points = []
    if not random_points:
        if min:
            if dim == 2:
                points = [
                    -4 * math.pi,
                    -8 / 3 * math.pi,
                    -2 / 3 * math.pi,
                    4 / 3 * math.pi,
                    10 / 3 * math.pi,
                    4 * math.pi,
                ]
            elif dim == 3:
                points = [(0.7, -0.7), (-0.7, 0.7)]
        elif not min:
            if dim == 2:
                points = [
                    4 * math.pi,
                    8 / 3 * math.pi,
                    2 / 3 * math.pi,
                    -4 / 3 * math.pi,
                    -10 / 3 * math.pi,
                    -4 * math.pi,
                ]
            elif dim == 3:
                points = [(-0.7, -0.7), (0.7, 0.7)]
    else:
        points.append(random.uniform(domain[0][0], domain[0][1]))
        if dim == 3:
            points.append(random.uniform(domain[1][0], domain[1][1]))
    return points


if __name__ == "__main__":

    values_test = False
    descent = True

    if not values_test:
        params = [(0.001, 5000)]
    else:
        params = generate_test_params()

    points = generate_points(True)

    print("=" * 100)

    for learning_rate, max_step_count in params:
        args, total_time = grad_descent(
            f,
            f_derivative,
            [(-4 * math.pi, 4 * math.pi)],
            learning_rate,
            [-8 / 3 * math.pi],
            max_step_count,
            find_min=descent,
            plot=True,
        )

        print(
            f"\nFunkcja f(x) : współczynnik długości kroku: {learning_rate}, max ilość kroków: {max_step_count}, czas: {total_time:.6f}s"
        )
        print(f"Punkt startowy: x = {points[0]}, y = {f(points[0])}")
        print(f"Punkt końcowy: x = {args[0]}, y = {f(args[0])}")

    points = generate_points(True, [(-2, 2), (-2, 2)], True, 3)
    print("=" * 100)
    # plt.show()

    for learning_rate, max_step_count in params:
        args, total_time = grad_descent(
            g,
            g_derivative,
            [(-2, 2), (-2, 2)],
            learning_rate,
            [points[0], points[1]],
            max_step_count,
            find_min=descent,
            plot=False,
        )
        print(
            f"\nFunkcja g(x,y) : długość kroku: {learning_rate}, max ilość kroków: {max_step_count}, czas: {total_time:.6f}s"
        )
        print(
            f"Punkt startowy: x = {points[0]}, y = {points[1]}, z = {g(points[0], points[1])}"
        )
        print(
            f"Punkt końcowy: x = {args[0]}, y = {args[1]}, z = {g(args[0], args[1])}\n"
        )

    print("=" * 100)
