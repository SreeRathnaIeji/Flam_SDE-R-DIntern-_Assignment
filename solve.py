import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

data = pd.read_csv("xy_data.csv")

x_expected = data["x"].to_numpy()
y_expected = data["y"].to_numpy()

def curve(theta, M, X, t):

    x = (
        t * np.cos(theta)
        - np.exp(M * np.abs(t))
        * np.sin(0.3 * t)
        * np.sin(theta)
        + X
    )

    y = (
        42
        + t * np.sin(theta)
        + np.exp(M * np.abs(t))
        * np.sin(0.3 * t)
        * np.cos(theta)
    )

    return x, y

def cost(parameters):

    theta, M, X = parameters

    t = np.linspace(6, 60, len(x_expected))

    x_predicted, y_predicted = curve(theta, M, X, t)

    error = np.sum(
        np.abs(x_expected - x_predicted)
        + np.abs(y_expected - y_predicted)
    )

    return error

result = differential_evolution(
    cost,
    bounds,
    seed=42
)

theta, M, X = result.x

print("Theta (degrees):", np.rad2deg(theta))
print("M:", M)
print("X:", X)
print("L1 cost:", result.fun)
