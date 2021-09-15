import numpy as np
from termcolor import colored as tc
import warnings
warnings.filterwarnings("error")


class Alg:
    def __init__(self, feature_dim, n_arms, features, true_theta):
        self.feature_dim = feature_dim
        self.n_arms = n_arms
        
        self.H = np.zeros(shape=(n_arms, 1, feature_dim), dtype=np.float64) # F.T * F
        self.Y = np.zeros(shape=(n_arms, 1), dtype=np.float64)              # F.T * r

        self.C = 1.00 * np.identity(n=feature_dim, dtype=np.float64)
        self.S = 1e+6 * np.identity(n=feature_dim, dtype=np.float64)
        self.S0 = 1e-2 * np.identity(n=feature_dim, dtype=np.float64)

        self.R = np.empty(shape=(n_arms, self.H.shape[1]), dtype=np.float64)
        self._idle(features, true_theta)

        self.P = np.empty(shape=(n_arms, feature_dim, feature_dim), dtype=np.float64)
        self.Theta = np.zeros(shape=(n_arms, feature_dim), dtype=np.float64)
        self.alpha = np.empty(shape=n_arms, dtype=np.float64)
        self.beta = np.empty_like(self.alpha)

        for arm in range(n_arms):
            self.P[arm] = self.S0 + (self.H[arm].T * self.R[arm]) @ self.H[arm]
            self.Theta[arm] = np.dot(np.linalg.inv(self.P[arm]),
                    self.H[arm].T * self.R[arm] * self.Y[arm])[:, 0]
            self.alpha[arm] = np.dot(self.R[arm].dot(self.Y[arm]), self.Y[arm])
            self.beta[arm] = 1 - self.alpha[arm] + np.dot(self.P[arm].dot(self.Theta[arm]),
                    self.Theta[arm])

            # sanity check
            assert 0 < self.beta[arm] < 1
            assert (np.linalg.norm(self.Theta[arm] - true_theta[arm]) < \
                    np.sqrt(self.beta[arm] * (1 / np.linalg.eig(self.P[arm])[0]).max()))

        self.p = np.empty_like(self.alpha)

        self.arm = 0
        self.t = 0
        self.err = None

    def choose_arm(self, features):
        for arm in range(self.n_arms):
            try:
                self.p[arm] = np.dot(features[arm], self.Theta[arm]) + \
                    np.sqrt(self.beta[arm] * np.dot(np.linalg.inv(self.P[arm]).dot(features[arm]),
                        features[arm]))
            except RuntimeWarning:
                print(self.beta[arm])
                print(np.dot(np.linalg.inv(self.P[arm]).dot(features[arm]),
                        features[arm]))
                
                exit()
        self.arm = np.argmax(self.p)
        self.t += 1

        return self.arm

    def update(self, chosen_arm, chosen_arm_features, reward, true_theta):
        self.H[chosen_arm] = chosen_arm_features
        self.Y[chosen_arm] = reward

        arm = chosen_arm
        B_prev = self.P[arm] + self.C.T @ self.S @ self.C
        P_prev_Theta = self.P[arm].dot(self.Theta[arm])
        B_prev_inv = np.linalg.inv(B_prev)

        self.alpha[arm] = self.alpha[arm] + np.dot(self.R[arm].dot(self.Y[arm]), self.Y[arm]) -\
                np.dot(P_prev_Theta, np.dot(B_prev_inv, P_prev_Theta))
        self.P[arm] = self.S + (self.H[arm].T * self.R[arm]) @ self.H[arm] -\
                self.S @ self.C @ B_prev_inv @ self.C.T @ self.S
        self.Theta[arm] = np.dot(
                np.linalg.inv(self.P[arm]), 
                (np.dot(self.S @ self.C, np.dot(B_prev_inv, P_prev_Theta)) +\
                        (self.H[arm].T * self.R[arm] * self.Y[arm])[:, 0]))
        self.beta[arm] = 1 - self.alpha[arm] + np.dot(self.P[arm].dot(self.Theta[arm]),
                self.Theta[arm])

        assert 0 < self.beta[arm] < 1
        # TODO: uncomment the assertion if necessary
        # assert (np.linalg.norm(self.Theta[arm] - true_theta[arm]) < \
                # np.sqrt(self.beta[arm] * (1 / np.linalg.eig(self.P[arm])[0]).max()))
        self.S *= 1.0001
    
    def _idle(self, features, true_theta):
        BEST_ARMS = [3, 7, 9, 15]
        for arm in range(self.n_arms):
            self.R[arm] = 1.0 * np.identity(n=self.R.shape[1], dtype=np.float64)
            if arm in BEST_ARMS:
                self.R[arm] *= 1.2

            self.Y[arm] = features[arm].dot(true_theta[arm]) + 0.01 * np.random.randn()
            self.H[arm] = features[arm]

            # sanity check
            x = self.Y[arm] - self.H[arm].dot(true_theta[arm])
            iTheta = np.dot(self.S0.dot(true_theta[arm]), true_theta[arm]) +\
                    np.dot(self.R[arm].dot(x), x)
            assert iTheta <= 1
