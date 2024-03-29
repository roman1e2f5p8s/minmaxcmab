import numpy as np
from termcolor import colored as tc
import warnings
from scipy.special import gamma as Gamma_func
warnings.filterwarnings("error")


class Alg:
    def __init__(self, n_trials, feature_dim, n_arms, features, true_theta, T):
        self.feature_dim = feature_dim
        self.n_arms = n_arms
        self.n_trials = n_trials
        self.T = T
        
        self.H = np.zeros(shape=(n_arms, 1, feature_dim), dtype=np.float64) # F.T * F
        self.Y = np.zeros(shape=(n_arms, 1), dtype=np.float64)              # F.T * r

        # TODO: change C, S, S0 here
        self.C = 1.00 * np.identity(n=feature_dim, dtype=np.float64)
        # S * sum_t||(Theta_{t+1} - Theta_t)||_2^2 <= 1
        # the more stationary environment, the larger S should be selected
        # BT = np.array([sum([np.linalg.norm(true_theta[t, a] - true_theta[t+1, a])**2 \
                # for t in range(n_trials-1)]) for a in range(n_arms)]).mean()
        # self.S = (1 / BT) * np.identity(n=feature_dim, dtype=np.float64)
        self.S = 10000 * np.identity(n=feature_dim, dtype=np.float64)
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
            self.beta[arm] = 1
            # self.beta[arm] = 1 - self.alpha[arm] + np.dot(self.P[arm].dot(self.Theta[arm]),
                    # self.Theta[arm])

            # sanity check
            # assert 0 < self.beta[arm] < 1
            # assert (np.linalg.norm(self.Theta[arm] - true_theta[0, arm]) < \
                    # np.sqrt(self.beta[arm] * (1 / np.linalg.eig(self.P[arm])[0]).max()))

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
                print('Negative beta:', self.beta[arm])
                print(np.dot(np.linalg.inv(self.P[arm]).dot(features[arm]),
                        features[arm]))
                exit()
        self.arm = np.argmax(self.p)
        self.t += 1

        return self.arm

    def update_nonstationary(self, chosen_arm, chosen_arm_features, reward, true_theta):
        # if self.t > 2000:
            # self.S = 1e6 * np.identity(n=self.feature_dim, dtype=np.float64)
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
        # self.beta[arm] = 1 - self.alpha[arm] + np.dot(self.P[arm].dot(self.Theta[arm]),
                # self.Theta[arm])
        self.beta[arm] = 1

        # assert 0 < self.beta[arm] <= 1
        # uncomment the assertion if necessary
        # assert (np.linalg.norm(self.Theta[arm] - true_theta[self.t-1, arm]) < \
                # np.sqrt(self.beta[arm] * (1 / np.linalg.eig(self.P[arm])[0]).max()))
        # self.S *= 1.0001
    
    # stationary update
    def update(self, chosen_arm, chosen_arm_features, reward, true_theta):
        self.H[chosen_arm] = chosen_arm_features
        self.Y[chosen_arm] = reward

        arm = chosen_arm
        P_prev_Theta = self.P[arm].dot(self.Theta[arm])
        
        # TODO: alpha is not updated!
        # self.alpha[arm] = self.alpha[arm] + np.dot(self.R[arm].dot(self.Y[arm]), self.Y[arm]) -\
                # np.dot(P_prev_Theta, np.dot(B_prev_inv, P_prev_Theta))
        P_prev = self.P[arm]
        self.P[arm] = self.S @ np.linalg.solve(self.P[arm] + self.S, self.P[arm]) + \
                (self.H[arm].T * self.R[arm]) @ self.H[arm] 
        self.Theta[arm] = (np.linalg.solve(self.P[arm], 
                self.S @ np.linalg.solve(P_prev + self.S, P_prev_Theta)) + \
            self.H[arm].T * self.R[arm] * self.Y[arm])[:, 0]
        # self.beta[arm] = 1 - self.alpha[arm] + np.dot(self.P[arm].dot(self.Theta[arm]),
                # self.Theta[arm])
        self.beta[arm] = 1

    
    def _idle(self, features, true_theta):
        # BEST_ARMS = [3, 7, 9, 15]
        c = 3
        x = 1 / (c * self.T)
        R_x = 1000
        print('R const', R_x * x)
        print('x', x)
        print('# TODO: alpha is not updated!')
        for arm in range(self.n_arms):
            # TODO: change R here
            self.R[arm] = R_x * x * np.identity(n=self.R.shape[1], dtype=np.float64)
            # if arm in BEST_ARMS:
                # self.R[arm] *= 1.2

            # self.Y[arm] = features[arm].dot(true_theta[0, arm]) + 0.01 * np.random.randn()
            self.Y[arm] = features[arm].dot(true_theta[0, arm]) + np.random.laplace(scale=1)
            self.H[arm] = features[arm]

            # sanity check
            # x = self.Y[arm] - self.H[arm].dot(true_theta[0, arm])
            # iTheta = np.dot(self.S0.dot(true_theta[0, arm]), true_theta[0, arm]) +\
                    # np.dot(self.R[arm].dot(x), x)
            # assert iTheta <= 1
