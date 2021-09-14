import numpy as np
from termcolor import colored as tc


class Alg:
    def __init__(self, feature_dim, n_arms, features, true_theta):
        self.feature_dim = feature_dim
        self.n_arms = n_arms
        
        self.H = np.zeros(shape=(n_arms, feature_dim, feature_dim), dtype=np.float64)   # F.T * F
        self.Y = np.zeros(shape=(n_arms, feature_dim), dtype=np.float64)                # F.T * r

        # TODO: in general, C, R, S can be different for each arm
        self.C = 1.00 * np.identity(n=feature_dim, dtype=np.float64)
        # self.R = 2.0e-1 * np.identity(n=feature_dim, dtype=np.float64)
        self.S = 1e+6 * np.identity(n=feature_dim, dtype=np.float64)
        self.S0 = 1e-2 * np.identity(n=feature_dim, dtype=np.float64)

        # self.R_inv = np.linalg.pinv(self.R)
        # self.S_inv = np.linalg.pinv(self.S)

        self.R = np.empty_like(self.H)
        self._idle(features, true_theta)

        self.P = np.empty_like(self.H)
        self.Theta = np.zeros_like(self.Y)
        self.alpha = np.empty(shape=n_arms, dtype=np.float64)
        self.beta = np.empty_like(self.alpha)

        for arm in range(n_arms):
            self.P[arm] = self.S0 + self.H[arm].T @ self.R[arm] @ self.H[arm]
            self.Theta[arm] = np.linalg.inv(self.P[arm]).dot(self.H[arm].T).dot(self.R[arm]).dot(self.Y[arm])
            self.alpha[arm] = np.dot(self.R[arm].dot(self.Y[arm]), self.Y[arm])
            self.beta[arm] = 1 - self.alpha[arm] + np.dot(self.P[arm].dot(self.Theta[arm]),
                    self.Theta[arm])

            # sanity check
            assert 0 < self.beta[arm] < 1
            # print('1-beta[{}] = {:.6f}'.format(arm, 1 - self.beta[arm]))
            assert (np.linalg.norm(self.Theta[arm] - true_theta[arm]) < \
                    np.sqrt(self.beta[arm] * (1 / np.linalg.eig(self.P[arm])[0]).max()))

        self.p = np.empty_like(self.alpha)

        self.arm = 0
        self.t = 0
        self.err = None

    def choose_arm(self, features):
        for arm in range(self.n_arms):
            self.p[arm] = np.dot(features[arm], self.Theta[arm]) + \
                    np.sqrt(self.beta[arm] * np.dot(np.linalg.inv(self.P[arm]).dot(features[arm]),
                        features[arm]))
        self.arm = np.argmax(self.p)
        self.t += 1

        return self.arm

    def update(self, chosen_arm, chosen_arm_features, reward, true_theta):
        self.H[chosen_arm] += np.outer(chosen_arm_features, chosen_arm_features)
        self.Y[chosen_arm] += reward * chosen_arm_features

        arm = chosen_arm
        B_prev = self.P[arm] + self.C.T @ self.S @ self.C
        P_prev_Theta = self.P[arm].dot(self.Theta[arm])
        B_prev_inv = np.linalg.inv(B_prev)

        self.alpha[arm] = self.alpha[arm] + np.dot(self.R[arm].dot(self.Y[arm]), self.Y[arm]) -\
                np.dot(P_prev_Theta, np.dot(B_prev_inv, P_prev_Theta))
        self.P[arm] = self.S + self.H[arm].T @ self.R[arm] @ self.H[arm] -\
                self.S @ self.C @ B_prev_inv @ self.C.T @ self.S
        self.Theta[arm] = np.dot(
                np.linalg.inv(self.P[arm]), 
                (np.dot(self.S @ self.C, np.dot(B_prev_inv, P_prev_Theta)) +\
                        np.dot(self.H[arm].T, self.R[arm].dot(self.Y[arm])))
            )
        self.beta[arm] = 1 - self.alpha[arm] + np.dot(self.P[arm].dot(self.Theta[arm]),
                self.Theta[arm])

        # print(self.beta[arm])
        assert 0 < self.beta[arm] < 1
        assert (np.linalg.norm(self.Theta[arm] - true_theta[arm]) < \
                np.sqrt(self.beta[arm] * (1 / np.linalg.eig(self.P[arm])[0]).max()))
        # self.err = np.sqrt(self.beta[arm] * (1 / np.linalg.eig(self.P[arm])[0]).max())
        self.err = self.beta
    
        if False:
            print('t={:6d} | played arm: {} |'.format(self.t, self.arm), end=' ')
            for arm in range(self.n_arms):
                # x = np.linalg.norm(self.Theta[arm])
                x = np.sqrt(self.beta[arm] * (1 / np.linalg.eig(self.P[arm])[0]).max())
                s = 'arm={}: {:.4f}'.format(arm, x)
                s = tc(s, 'green') if arm == self.arm else s
                print('{} |'.format(s), end=' ')
            print()

    def _idle(self, features, true_theta):
        BEST_ARMS = [3, 7, 9, 15]
        for arm in range(self.n_arms):
            self.R[arm] = 2.0e-1 * np.identity(n=self.feature_dim, dtype=np.float64)
            if arm in BEST_ARMS:
                self.R[arm] *= 5

            self.Y[arm] = (features[arm].dot(true_theta[arm]) + 0.01 * np.random.randn()) * features[arm]
            self.H[arm] = np.outer(features[arm], features[arm])

            # sanity check
            x = self.Y[arm] - self.H[arm].dot(true_theta[arm])
            iTheta = np.dot(self.S0.dot(true_theta[arm]), true_theta[arm]) +\
                    np.dot(self.R[arm].dot(x), x)
            # print('iTheta[{}] = {:.6f}'.format(arm, iTheta))
            assert iTheta <= 1
