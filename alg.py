import numpy as np


class Alg:
    def __init__(self, feature_dim, n_arms, alpha):
        self.feature_dim = feature_dim
        self.n_arms = n_arms
        
        self.H = np.zeros(shape=(n_arms, feature_dim, feature_dim), dtype=np.float64)   # F.T * F
        self.Y = np.zeros(shape=(n_arms, feature_dim), dtype=np.float64)                # F.T * R

        self.C = np.empty_like(self.H)
        self.R = np.empty_like(self.H)
        self.S = np.empty_like(self.H)
        self.P_inv = np.empty_like(self.H)
        self.V = np.empty_like(self.H)
        self.Theta = np.zeros_like(self.Y)

        for arm in range(n_arms):
            self.C[arm] = 0.5 * np.identity(n=feature_dim, dtype=np.float64)
            self.R[arm] = 10 * np.identity(n=feature_dim, dtype=np.float64)
            self.S[arm] = 1.0 * np.identity(n=feature_dim, dtype=np.float64)
            # self.P_inv[arm] = np.linalg.inv(self.S[arm] + self.H[arm].T @ self.R[arm] @ self.H[arm])
            # self.Theta[arm] = self.P_inv[arm].dot(self.H[arm].T).dot(self.R[arm]).dot(self.Y[arm])
            self.P_inv[arm] = 1 / self.S[arm] # assuming H[arm] is zeros and S[arm] is diagonal
            # self.Theta[arm] = np.zeros(feature_dim) # assuming H[arm] is zeros
            self.V[arm] = 1 / self.S[arm] + self.C[arm] @ self.P_inv[arm] @ self.C[arm].T

        self.p = np.empty(shape=n_arms, dtype=np.float64)


        # LinUCB
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
        self.H[chosen_arm] += np.outer(chosen_arm_features, chosen_arm_features)
        self.Y[chosen_arm] += reward * chosen_arm_features

        # LinUCB
        self.A[chosen_arm] += np.outer(chosen_arm_features, chosen_arm_features)
        self.b[chosen_arm] += reward * chosen_arm_features
