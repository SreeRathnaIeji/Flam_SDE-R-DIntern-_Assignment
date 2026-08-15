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
