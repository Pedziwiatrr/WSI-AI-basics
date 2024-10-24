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
    step_count=100,
    find_min=True,
    plot=True,
):
    if find_min:
        learning_rate = -learning_rate
        # domyślnie dążymy do minimum (ponieważ algorytm to gradient descend).
        # Jeśli find_min = False, funkcja będzie dążyć do maksimum

    path = [args]

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
    if plot:
        if len(args) == 1:
            two_dimensions_chart(function, domain[0], path, (learning_rate, step_count))
        elif len(args) == 2:
            three_dimensions_chart(function, domain, path, (learning_rate, step_count))

    return args


def two_dimensions_chart(function, domain, path, gradient_params):
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
    plt.suptitle(
        f"{function.__name__}(x) / współczynnik długości kroku = {gradient_params[0]}, ilość kroków = {gradient_params[1]}",
        fontsize=12,
        color="blue",
    )
    plt.savefig(
        "./Zadanie1/wykresy/gradient_wykres_2d.png", dpi=500, bbox_inches="tight"
    )
    plt.show()


def three_dimensions_chart(function, domain, path, gradient_params):
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
    plt.savefig(
        "./Zadanie1/wykresy/gradient_wykres_3d.png", dpi=500, bbox_inches="tight"
    )
    plt.show()


def generate_test_params():
    learning_rates = [0.1, 0.01, 0.001]
    max_step_counts = [100, 500, 1000]
    test_params = []
    for lr in learning_rates:
        for msc in max_step_counts:
            test_params.append((lr, msc))
    return test_params


if __name__ == "__main__":
    random_x = random.uniform(-4 * math.pi, 4 * math.pi)
    values_test = True
    plot = True

    if not values_test:
        params = 0.05, 1000
    else:
        params = generate_test_params()

    for learning_rate, max_step_count in params:
        args = grad_descent(
            f,
            f_derivative,
            [(-4 * math.pi, 4 * math.pi)],
            learning_rate,
            [random_x],
            max_step_count,
            plot,
        )

        print(
            f"\nFunkcja f(x) : współczynnik długości kroku: {learning_rate}, ilość kroków: {max_step_count}"
        )
        print(f"Punkt startowy: x = {random_x}, y = {f(random_x)}")
        print(f"Punkt końcowy: x = {args[0]}, y = {f(args[0])}")

    random_x = random.uniform(-2, 2)
    random_y = random.uniform(-2, 2)

    for learning_rate, max_step_count in params:
        args = grad_descent(
            g,
            g_derivative,
            [(-2, 2), (-2, 2)],
            learning_rate,
            [random_x, random_y],
            max_step_count,
        )
        print(
            f"\nFunkcja g(x,y) : długość kroku: {learning_rate}, ilość kroków: {max_step_count}"
        )
        print(
            f"\nPunkt startowy: x = {random_x}, y = {random_y}, z = {g(random_x, random_y)}"
        )
        print(
            f"Punkt końcowy: x = {args[0]}, y = {args[1]}, z = {g(args[0], args[1])}\n"
        )
