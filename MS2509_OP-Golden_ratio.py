
# Objective function
# ---------------------------------------------------------
def f(x):
    return x**3 - 3*x**2 + 7

# Golden Section Search Algorithm
def golden_section_search(func, a, b, tol=1e-6, max_iter=200):
    """
    Finds the x that minimizes func(x) over [a, b]
    using the Golden Section Search method.

    Parameters
    ----------
    func     : objective function to minimize
    a, b     : lower and upper bounds of the search interval
    tol      : tolerance for stopping (width of interval)
    max_iter : maximum number of iterations (safety limit)

    Returns
    -------
    x_opt : approximate optimal x
    f_opt : function value at x_opt
    """

    # Golden ratio constant
    gr = (5 ** 0.5 - 1) / 2      # gr = 0.618033988...

    # Initial two interior points
    x1 = b - gr * (b - a)
    x2 = a + gr * (b - a)
    f1 = func(x1)
    f2 = func(x2)

    iteration = 0
    print(f"{'Iter':<5}{'a':<12}{'x1':<12}{'x2':<12}{'b':<12}{'f(x1)':<12}{'f(x2)':<12}")

    while abs(b - a) > tol and iteration < max_iter:
        print(f"{iteration:<5}{a:<12.6f}{x1:<12.6f}{x2:<12.6f}{b:<12.6f}{f1:<12.6f}{f2:<12.6f}")

        if f1 < f2:
            # Minimum lies in [a, x2]
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - gr * (b - a)
            f1 = func(x1)
        else:
            # Minimum lies in [x1, b]
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + gr * (b - a)
            f2 = func(x2)

        iteration += 1

    x_opt = (a + b) / 2
    f_opt = func(x_opt)
    return x_opt, f_opt, iteration

# Main Program
if __name__ == "__main__":
    a, b = 1, 3          # Interval [1, 3]
    tolerance = 1e-6

    x_opt, f_opt, n_iter = golden_section_search(f, a, b, tolerance)

    print("\n---------------------------------------------")
    print(f"Optimal x  = {x_opt:.6f}")
    print(f"Minimum f(x) = {f_opt:.6f}")
    print(f"Number of iterations = {n_iter}")
    print("---------------------------------------------")
