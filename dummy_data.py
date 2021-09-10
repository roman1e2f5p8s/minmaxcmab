from numpy import array
from scipy.io import savemat
# T = 10, A = 2, D = 3

DATA = array([
        [[0.1, 0.2, 0.3],
         [0.4, 0.5, 0.6]],
        [[0.2, 0.3, 0.1],
         [0.5, 0.6, 0.4]],
        [[0.3, 0.1, 0.2],
         [0.6, 0.4, 0.5]],
        [[0.7, 0.8, 0.9],
         [0.1, 0.2, 0.3]],
        [[0.8, 0.9, 0.7],
         [0.2, 0.3, 0.1]],
        [[0.9, 0.7, 0.8],
         [0.3, 0.1, 0.2]],
        [[0.3, 0.4, 0.5],
         [0.9, 0.0, 0.1]],
        [[0.4, 0.5, 0.3],
         [0.0, 0.1, 0.9]],
        [[0.5, 0.3, 0.4],
         [0.1, 0.9, 0.0]],
        [[0.3, 0.2, 0.1],
         [0.6, 0.5, 0.4]]
])

TRUE_THETA = array([
    [0.15, 0.46, 0.21],
    [0.72, 0.08, 0.64]
])

NOISE = array([
    0.004,
    0.003,
    0.002,
    0.001,
    0.009,
    0.003,
    0.001,
    0.002,
    0.005,
    0.007,
    ])

savemat('dummy_data.mat', {'features': DATA, 'true_theta': TRUE_THETA, 'noise': NOISE})
