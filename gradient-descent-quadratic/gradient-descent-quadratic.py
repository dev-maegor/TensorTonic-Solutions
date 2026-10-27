def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    x = float(x0)

    for _ in range(steps):
        gradient = 2 * a * x + b
        x -= lr * gradient

    return float(x)