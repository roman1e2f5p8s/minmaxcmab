import numpy as np


class LinUCB:
    def __init__(self, feature_dim, n_arms, alpha):
        self.feature_dim = feature_dim  # d in the paper
        self.n_arms = n_arms  # A_t in the paper
        self.alpha = alpha

        self.b = np.zeros(shape=(n_arms, feature_dim), dtype=np.float64)
        self.A = np.empty(shape=(n_arms, feature_dim, feature_dim), dtype=np.float64)
        for arm in range(n_arms):
            self.A[arm] = np.identity(n=feature_dim, dtype=np.float64)
        self.p = np.empty(shape=n_arms, dtype=np.float64)

    def choose_arm(self, features):
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            self.p[arm] = np.dot(np.dot(A_inv, self.b[arm]).T, features[arm]) + \
                    self.alpha * np.sqrt(np.dot(features[arm].T, np.dot(A_inv, features[arm])))

        return np.argmax(self.p)

    def update(self, chosen_arm, chosen_arm_features, reward):
        self.A[chosen_arm] += np.outer(chosen_arm_features, chosen_arm_features)
        self.b[chosen_arm] += reward * chosen_arm_features
