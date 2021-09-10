import numpy as np


class Alg:
    def __init__(self, feature_dim, n_arms):
        self.feature_dim = feature_dim
        self.n_arms = n_arms
        
        self.H = np.zeros(shape=(n_arms, feature_dim, feature_dim), dtype=np.float64)   # F.T * F
        self.Y = np.zeros(shape=(n_arms, feature_dim), dtype=np.float64)                # F.T * R

        # TODO: in general, C, R, S can be different for each arm
        self.C = 1 * np.identity(n=feature_dim, dtype=np.float64)
        self.R = 1 * np.identity(n=feature_dim, dtype=np.float64)
        self.S = 1 * np.identity(n=feature_dim, dtype=np.float64)
        self.R_inv = np.linalg.inv(self.R)
        self.S_inv = np.linalg.inv(self.S)

        self.P_inv = np.empty_like(self.H)
        self.Theta = np.zeros_like(self.Y)
        self.alpha = np.zeros(shape=n_arms, dtype=np.float64)
        self.beta = np.ones(shape=n_arms, dtype=np.float64)
        self.arm = 0
        self.t = 0

        for arm in range(n_arms):
            # self.P_inv[arm] = np.linalg.inv(self.S[arm] + self.H[arm].T @ self.R[arm] @ self.H[arm])
            # self.Theta[arm] = self.P_inv[arm].dot(self.H[arm].T).dot(self.R[arm]).dot(self.Y[arm])
            self.P_inv[arm] = self.S_inv # assuming H[arm] is zeros
            # self.alpha[arm] = self.R.dot(self.Y[arm]).dot(self.Y[arm]) # TODO: only if Y == zeros!

        self.p = np.empty(shape=n_arms, dtype=np.float64)

    def choose_arm(self, features):
        for arm in range(self.n_arms):
            V = self.S_inv + self.C @ self.P_inv[arm] @ self.C.T

            self.P_inv[arm] = V - V @ self.H[arm].T @ \
                    np.linalg.inv(self.R_inv + self.H[arm] @ V @ self.H[arm].T) @ \
                    self.H[arm] @ V

            self.Theta[arm] = self.P_inv[arm].dot(np.linalg.inv(V)).\
                    dot(self.C).dot(self.Theta[arm]) + \
                    self.P_inv[arm].dot(self.H[arm].T).dot(self.R).dot(self.Y[arm])

            self.p[arm] = np.dot(features[arm], self.Theta[arm]) + \
                    np.sqrt(self.beta[arm] * np.dot(self.P_inv[arm].dot(features[arm]), features[arm]))

        self.arm = np.argmax(self.p)
        self.t += 1

        return self.arm

    def update(self, chosen_arm, chosen_arm_features, reward):
        self._calculate_beta(chosen_arm)
        self.H[chosen_arm] += np.outer(chosen_arm_features, chosen_arm_features)
        self.Y[chosen_arm] += reward * chosen_arm_features

    def _calculate_beta(self, arm):
        x = self.R.dot(self.Y[arm]).dot(self.Y[arm])
        y = np.dot(np.linalg.inv(self.P_inv[arm] + \
                    self.P_inv[arm] @ self.C.T @ self.S @ self.C @ self.P_inv[arm]
                    ).dot(self.Theta[arm]), self.Theta[arm])
        z = np.linalg.inv(self.P_inv[arm]).dot(self.Theta[arm]).dot(self.Theta[arm])
        # print(self.Theta[arm])
        # print(np.linalg.eigvals(self.P_inv[arm]))
        # print(z)

        self.alpha[arm] += (x - y)
        self.beta[arm] = 1 - self.alpha[arm] + z

        if self.beta[arm] < 0:
            print('Negative beta for arm {} at time {}'.format(arm, self.t))
            exit()
