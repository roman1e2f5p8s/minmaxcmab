import numpy as np


class BMM:
    def __init__(self, feature_dim: int, n_arms: int, eps: float, delta: float, v: float, T: int):
        assert (feature_dim > 0 and type(feature_dim) == int)
        self.feature_dim = feature_dim  # d in the paper

        assert (n_arms > 0 and type(n_arms) == int)
        self.n_arms = n_arms  # K in the paper

        assert 0 < eps <= 1
        self.eps = eps
        self.v = v

        assert 0 < delta <= 1
        self.delta = delta
        self.r = int(np.ceil(8 * np.log(2 * n_arms * T * np.log(T) / delta)))
        
        assert (T > 0 and type(T) == int)
        self.T = T

        self.arms = np.empty(shape=T, dtype=np.int32)
        
        # BMM requires storing contexts x for each time t and each arm a 
        self.x = np.zeros(shape=(T, n_arms, feature_dim), dtype=np.float64)

        self.A_inv = None

        self.reward = np.zeros(shape=(self.r, T, n_arms), dtype=np.float64)

    def _alpha(self):
        return np.power(12 * self.v, 1 / (1 + self.eps)) * np.power(self.t + 1, 0.5 * (1 - self.eps) /\
                (1 + self.eps))

    def step(self, t, rewards, features, Psi):
        # rewards --- self.r rewards received when playing choose_arm arm self.r times
        A = np.identity(n=self.feature_dim, dtype=np.float64)

        Theta_hat = np.zeros(shape=(self.r, self.feature_dim), dtype=np.float64)

        for j in range(self.r):
            b = np.zeros(shape=(self.r, self.feature_dim), dtype=np.float64)
            for tau in Psi:
                if not j:
                    A += self.x[tau-1, self.arms[tau-1]] @ self.x[tau-1, self.arms[tau-1]].T
                if tau == Psi[0]:
                    self.reward[j, t, self.arms[t]] = rewards[j]
                b[j] += self.reward[j, tau-1, self.arms[tau-1]] * self.x[tau-1, self.arms[tau-1]]
            if not j:
                self.A_inv = np.linalg.inv(A)
            Theta_hat[j] = np.dot(self.A_inv, b[j])

        r_hat = np.empty(shape=self.n_arms, dtype=np.float64)
        w = np.empty(shape=self.n_arms, dtype=np.float64)

        for i, arm in enumerate(self.n_arms):
            self.x[t, arm] = features[arm]

            r_hat[i] = np.median(np.array(
                    [np.dot(self.x[t, arm].T, Theta_hat[j]) for j in range(self.r)],
                    dtype=np.float64,
                    ))

            w[i] = (1 + self._alpha()) * np.sqrt(
                    np.dot(
                        self.x[t, arm].T,
                        np.dot(
                            self.A_inv,
                            self.x[t, arm]
                            )
                        )
                    )

        return r_hat, w

    def play_arm(self, t, arm):
        self.arms[t] = arm
