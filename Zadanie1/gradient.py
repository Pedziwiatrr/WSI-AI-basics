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
    all_ex=False,
    ax=None,
):
    if find_min:
        learning_rate = -learning_rate
        # domyślnie dążymy do minimum (ponieważ algorytm to gradient descent).
        # Jeśli find_min = False, funkcja będzie dążyć do maksimum

    path = [args]
    start_time = time.time()

    for i in range(step_count):
        new_args = [
            variable + learning_rate * derivative(*args)[i]
            for i, variable in enumerate(args)
        ]

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
                function,
                domain[0],
                path,
                (learning_rate, step_count),
                total_time,
                all_ex,
            )
        elif len(args) == 2:
            three_dimensions_chart(
                function,
                domain,
                path,
                (learning_rate, step_count),
                total_time,
                all_ex,
                ax,
            )

    return (args, total_time)


def two_dimensions_chart(function, domain, path, gradient_params, time, all_ex):
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

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.suptitle(
        f"{function.__name__}(x) / współczynnik długości kroku = {gradient_params[0]}, max ilość kroków = {gradient_params[1]},\n czas = {time:.6f}s",
        fontsize=12,
        color="blue",
    )
    if not all_ex:
        plt.legend()
        plt.show()
    plt.savefig(
        f"./Zadanie1/wykresy/f/f_{gradient_params[0]}_{gradient_params[1]}.png",
        dpi=500,
        bbox_inches="tight",
    )


def three_dimensions_chart(
    function, domain, path, gradient_params, time, all_ex, ax=None
):
    if not ax:
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
    plt.suptitle(
        f"{function.__name__}(x, y) / współczynnik długości kroku = {gradient_params[0]}, max ilość kroków = {gradient_params[1]},\n czas = {time:.6f}s",
        fontsize=12,
        color="blue",
    )
    if not all_ex:
        plt.legend()
        plt.show()
    plt.savefig(
        f"./Zadanie1/wykresy/g/g_{gradient_params[0]}_{gradient_params[1]}.png",
        dpi=500,
        bbox_inches="tight",
    )


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
                    -3.5 * math.pi,
                    -7 / 3 * math.pi,
                    0,
                    5 / 6 * math.pi,
                    11 / 3 * math.pi,
                ]
            elif dim == 3:
                points = [(1, 0), (-0.6, 2)]
        elif not min:
            if dim == 2:
                points = [
                    3.5 * math.pi,
                    5 / 3 * math.pi,
                    0,
                    -math.pi,
                    -9 / 3 * math.pi,
                ]
            elif dim == 3:
                points = [(-0.5, -0.8), (1.5, 0.9)]
    else:
        random_x = random.uniform(domain[0][0], domain[0][1])
        if dim == 3:
            random_y = random.uniform(domain[1][0], domain[1][1])
            points.append((random_x, random_y))
        else:
            points.append(random_x)
    return points


if __name__ == "__main__":
    show_all_ex = True
    values_test = False
    descent = True

    if not values_test:
        params = [(0.001, 5000)]
    else:
        params = generate_test_params()

    points = generate_points(not show_all_ex, min=descent)

    print("=" * 100)

    if show_all_ex:
        for point in points:
            args, total_time = grad_descent(
                f,
                f_derivative,
                [(-4 * math.pi, 4 * math.pi)],
                params[0][0],
                [point],
                params[0][1],
                find_min=descent,
                plot=True,
                all_ex=True,
            )
            print(f"Punkt końcowy: x = {args[0]}, y = {f(args[0])}")
    else:
        for learning_rate, max_step_count in params:
            args, total_time = grad_descent(
                f,
                f_derivative,
                [(-4 * math.pi, 4 * math.pi)],
                learning_rate,
                [points[0]],
                max_step_count,
                find_min=descent,
                plot=True,
            )

            print(
                f"\nFunkcja f(x) : współczynnik długości kroku: {learning_rate}, max ilość kroków: {max_step_count}, czas: {total_time:.6f}s"
            )
            print(f"Punkt startowy: x = {points[0]}, y = {f(points[0])}")
            print(f"Punkt końcowy: x = {args[0]}, y = {f(args[0])}")

    plt.show()
    points = generate_points(not show_all_ex, [(-2, 2), (-2, 2)], descent, 3)
    print("=" * 100)

    if show_all_ex:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        # tworzymy wykres by nie resetował się co iteracje
        for point in points:
            args, total_time = grad_descent(
                g,
                g_derivative,
                [(-2, 2), (-2, 2)],
                params[0][0],
                [point[0], point[1]],
                params[0][1],
                find_min=descent,
                plot=True,
                all_ex=True,
                ax=ax,
            )
            print(
                f"Punkt końcowy: x = {args[0]}, y = {args[1]}, z = {g(args[0], args[1])}"
            )
    else:
        for learning_rate, max_step_count in params:
            args, total_time = grad_descent(
                g,
                g_derivative,
                [(-2, 2), (-2, 2)],
                learning_rate,
                [points[0][0], points[0][1]],
                max_step_count,
                find_min=descent,
                plot=True,
            )
            print(
                f"\nFunkcja g(x,y) : długość kroku: {learning_rate}, max ilość kroków: {max_step_count}, czas: {total_time:.6f}s"
            )
            print(
                f"Punkt startowy: x = {points[0]}, y = {points[1]}, z = {g(points[0][0], points[0][1])}"
            )
            print(
                f"Punkt końcowy: x = {args[0]}, y = {args[1]}, z = {g(args[0], args[1])}"
            )

    print("=" * 100)
    plt.show()
