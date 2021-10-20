# D-LinUCB joint version
import numpy as np


class DLinUCBJ:
    def __init__(self, feature_dim, n_arms, delta, sigma, lambda_, L, S, gamma):
        '''
        feature_dim: dimension of a feature, feature_dim >= 1
        n_arms: the number of arms, n_arms >= 1
        delta: probability, 0 < delta <= 1 (should be low)
        sigma: subgaussianity constant
        lambda_: regularization, lambda_ > 0
        L: upper bound for actions
        S: upper bound for parameters
        gamma: the discount factor, 0 < gamma < 1
        '''
        
        assert feature_dim >= 1
        assert n_arms >= 1
        assert 0 < delta <= 1
        assert 0 < gamma < 1

        self.feature_dim = feature_dim
        self.n_arms = n_arms
        self.delta = delta
        self.sigma = sigma
        self.lambda_ = lambda_
        self.L = L
        self.S = S
        self.gamma = gamma

        self.b = np.zeros(shape=feature_dim, dtype=np.float64)
        self.V = lambda_ * np.identity(n=feature_dim)
        self.V_tilde = lambda_ * np.identity(n=feature_dim)
        self.theta_hat = np.zeros(shape=feature_dim, dtype=np.float64)

        # some constant expressions
        self.gamma2 = np.power(gamma, 2)
        self.sqrt_lambda_x_S = np.sqrt(lambda_) * S
        self.m2_log_delta = 2 * np.log(1 / delta)
        self.lambda_d_gamma = lambda_ * feature_dim * (1 - self.gamma2)
        self.gamma_lambda = (1 - gamma) * lambda_
        self.gamma2_lambda = (1 - self.gamma2) * lambda_

        self.ucb = np.empty(shape=n_arms, dtype=np.float64)

    def __repr__(self):
        return 'D-LinUCB algorithm with parameters:\n' + \
                '\t- delta={}\n'.format(self.delta) + \
                '\t- sigma={}\n'.format(self.sigma) + \
                '\t- lambda={}\n'.format(self.lambda_) + \
                '\t- L={}\n'.format(self.L) + \
                '\t- S={}\n'.format(self.S) + \
                '\t- gamma={}\n'.format(self.gamma)

    def choose_arm(self, t, features):
        '''
        t: time step (trial), t >= 0
        features: observed featured for all arms at time t
        '''
        # TODO: check which log is meant
        beta = self.sqrt_lambda_x_S + self.sigma * np.sqrt(
                self.m2_log_delta + self.feature_dim * np.log(
                    1 + np.power(self.L, 2) * (1 - np.power(self.gamma2, t)) / self.lambda_d_gamma
                    )
                )
        V_inv = np.linalg.inv(self.V)
        VVV = V_inv @ self.V_tilde @ V_inv

        for arm in range(self.n_arms):
            self.ucb[arm] = np.dot(features[arm].T, self.theta_hat) + \
                    beta * np.sqrt(np.dot(features[arm].T, VVV.dot(features[arm])))

        return np.argmax(self.ucb)

    def update(self, chosen_arm, chosen_arm_features, reward):
        out = np.outer(chosen_arm_features, chosen_arm_features)
        self.V = self.gamma*self.V + out + self.gamma_lambda*np.identity(self.feature_dim)
        self.V_tilde = self.gamma2*self.V_tilde + out + self.gamma2_lambda*np.identity(self.feature_dim)
        self.b = self.gamma + reward * chosen_arm_features
        self.theta_hat = np.linalg.inv(self.V).dot(self.b)
