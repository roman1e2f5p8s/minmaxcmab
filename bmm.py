import numpy as np


class BMM:
    def __init__(self, feature_dim: int, n_arms: int, eps: float, delta: float, v: float, T: int,
            r: int):
        assert (feature_dim > 0 and type(feature_dim) == int)
        self.feature_dim = feature_dim  # d in the paper

        assert (n_arms > 0 and type(n_arms) == int)
        self.n_arms = n_arms  # K in the paper

        assert 0 < eps <= 1
        self.eps = eps
        self.v = v

        assert 0 < delta <= 1
        self.delta = delta

        self.r = r
        
        assert (T > 0 and type(T) == int)
        self.T = T
        self.t = 0

        # self.arms = np.empty(shape=0, dtype=np.int32) # updated in SupBMM
        self.x = np.zeros(shape=(0, feature_dim), dtype=np.float64) # updated in SupBMM

    def _alpha(self):
        '''
        '''
        return np.power(12 * self.v, 1 / (1 + self.eps)) * np.power(self.t + 1, 0.5 * (1 - self.eps) /\
                (1 + self.eps))

    def step(self, rewards, features, Psi):
        '''
        Inputs:
            - rewards: self.r rewards received when playing choose_arm arm self.r times
            - features: array of shape (n_arms, feature_dim)
            - Psi: vector of length up to t, contains indices starting from zero
        Returns:
            - r_hat: vector of length self.n_arms
            - w: vector of length self.n_arms
        '''
        A = np.identity(n=self.feature_dim, dtype=np.float64)
        A_inv = np.identity(n=self.feature_dim, dtype=np.float64)

        b = np.zeros(shape=(self.r, self.feature_dim), dtype=np.float64)
        Theta_hat = np.zeros(shape=(self.r, self.feature_dim), dtype=np.float64)

        for j in range(self.r):
            for tau in Psi:
                if not j:
                    A += self.x[tau] @ self.x[tau].T
                b[j] += rewards[j] * self.x[tau]
            if not j:
                A_inv = np.linalg.inv(A)
            Theta_hat[j] = np.dot(A_inv, b[j])

        r_hat = np.empty(shape=self.n_arms, dtype=np.float64)
        w = np.empty(shape=self.n_arms, dtype=np.float64)

        for arm in range(self.n_arms):
            r_hat[arm] = np.median(
                    np.array(
                        [np.dot(features[arm].T, Theta_hat[j]) for j in range(self.r)],
                        dtype=np.float64,
                    ))

            w[arm] = (1 + self._alpha()) * np.sqrt(
                    np.dot(features[arm].T, np.dot(A_inv, features[arm]))
                    )
        self.t += 1

        return r_hat, w
