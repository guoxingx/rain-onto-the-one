
import numpy as np


def mock_slm_nd(size):
    # pseudothermal light in Gaussian distribution
    # light = np.random.normal(0, 1, size)
    light = np.random.uniform(0, 255, size)
    return np.zeros(size) + light


if __name__ == "__main__":
    pass
