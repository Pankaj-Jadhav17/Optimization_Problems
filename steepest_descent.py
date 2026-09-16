"""Steepest-descent minimization with golden-section line search."""

import numpy as np


def golden_section_search(func, lower, upper, tolerance=1e-8, max_iter=200):
    """Minimize a unimodal scalar function on [lower, upper]."""
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    x_left = upper - ratio * (upper - lower)
    x_right = lower + ratio * (upper - lower)
    f_left = func(x_left)
    f_right = func(x_right)

    for _ in range(max_iter):
        if abs(upper - lower) <= tolerance:
            break
        if f_left <= f_right:
            upper = x_right
            x_right = x_left
            f_right = f_left
            x_left = upper - ratio * (upper - lower)
            f_left = func(x_left)
        else:
            lower = x_left
            x_left = x_right
            f_left = f_right
            x_right = lower + ratio * (upper - lower)
            f_right = func(x_right)

    return (lower + upper) / 2.0


def numerical_gradient(func, x, step=1e-6):
    """Approximate the gradient of func at the vector x by central differences."""
    gradient = np.zeros_like(x, dtype=float)
    for index in range(x.size):
        offset = np.zeros_like(x, dtype=float)
        offset[index] = step
        gradient[index] = (
            func(x + offset) - func(x - offset)
        ) / (2.0 * step)
    return gradient


def steepest_descent(
    func,
    initial_x,
    tolerance=1e-6,
    gradient_step=1e-6,
    line_search_interval=(0.0, 10.0),
    max_iter=1000,
):
    """Minimize a differentiable function using normalized steepest descent.

    The stopping test follows the supplied algorithm:
    ||x(k+1) - x(k)|| / max(1, ||x(k)||) <= tolerance.
    """
    x = np.asarray(initial_x, dtype=float).reshape(-1)
    history = [x.copy()]

    for iteration in range(max_iter):
        gradient = numerical_gradient(func, x, gradient_step)
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm == 0.0:
            return x, func(x), iteration, history

        direction = -gradient / gradient_norm
        line_function = lambda alpha: func(x + alpha * direction)
        alpha = golden_section_search(
            line_function,
            line_search_interval[0],
            line_search_interval[1],
        )
        next_x = x + alpha * direction
        history.append(next_x.copy())

        relative_step = np.linalg.norm(next_x - x) / max(1.0, np.linalg.norm(x))
        x = next_x
        if relative_step <= tolerance:
            return x, func(x), iteration + 1, history

    return x, func(x), max_iter, history


if __name__ == "__main__":
    def objective(x):
        return (x[0] - 3.0) ** 2 + 2.0 * (x[1] + 1.0) ** 2


    minimizer, minimum, iterations, _ = steepest_descent(
        objective,
        initial_x=[0.0, 0.0],
    )
    print(f"Minimizer: {minimizer}")
    print(f"Minimum value: {minimum:.8f}")
    print(f"Iterations: {iterations}")