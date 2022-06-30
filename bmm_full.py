import numpy as np


class Bmm:
    def __init__(self, feature_dim: int, n_arms: int, eps: float, delta: float, v: float, T: int):
    # def __init__(self, feature_dim, n_arms, alpha):
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
        self.r = 5
        
        assert (T > 0 and type(T) == int)
        self.T = T

        self.t = 0
        self.Psi = np.empty(shape=0, dtype=np.int32)

        self.arms = np.empty(shape=T, dtype=np.int32)
        
        # BMM requires storing contexts x for each time t and each arm a 
        self.x = np.zeros(shape=(T, n_arms, feature_dim), dtype=np.float64)

        self.A = np.identity(n=feature_dim, dtype=np.float64)
        self.A_inv = np.linalg.inv(self.A)

        self.b = np.zeros(shape=(self.r, feature_dim), dtype=np.float64)

        self.reward = np.zeros(shape=(self.r, T, n_arms), dtype=np.float64)

        self.Theta_hat = np.zeros_like(self.b, dtype=np.float64)

        # oracle
        self.p = np.empty(shape=n_arms, dtype=np.float64)

        # self.b = np.zeros(shape=(n_arms, feature_dim), dtype=np.float64)
        # self.A = np.empty(shape=(n_arms, feature_dim, feature_dim), dtype=np.float64)
        # for arm in range(n_arms):
            # self.A[arm] = np.identity(n=feature_dim, dtype=np.float64)
        # self.p = np.empty(shape=n_arms, dtype=np.float64)
    
    def _alpha(self):
        return np.power(12 * self.v, 1 / (1 + self.eps)) * np.power(self.t + 1, 0.5 * (1 - self.eps) /\
                (1 + self.eps))

    def choose_arm(self, features):
        # self.t starts from 0
        for arm in range(self.n_arms):
            self.x[self.t, arm] = features[arm]

            r_hat = np.median(np.array(
                    [np.dot(self.x[self.t, arm].T, self.Theta_hat[j]) for j in range(self.r)],
                    dtype=np.float64,
                    ))

            w = (1 + self._alpha()) * np.sqrt(
                    np.dot(
                        self.x[self.t, arm].T,
                        np.dot(
                            self.A_inv,
                            self.x[self.t, arm]
                            )
                        )
                    )
            
            self.p[arm] = r_hat + w

            # A_inv = np.linalg.inv(self.A[arm])
            # self.p[arm] = np.dot(np.dot(A_inv, self.b[arm]).T, features[arm]) + \
                    # self.alpha * np.sqrt(np.dot(features[arm].T, np.dot(A_inv, features[arm])))

        self.arms[self.t] = np.argmax(self.p)

        return self.arms[self.t]

    def update(self, rewards):
        # rewards --- self.r rewards received when playing choose_arm arm self.r times
        self.A = np.identity(n=self.feature_dim, dtype=np.float64)

        self.reward[:, self.t, self.arms[self.t]] = rewards

        for j in range(self.r):
            self.b = np.zeros(shape=(self.r, self.feature_dim), dtype=np.float64)
            for tau in self.Psi:
                if not j:
                    self.A += self.x[tau-1, self.arms[tau-1]] @ self.x[tau-1, self.arms[tau-1]].T
                    # self.A += np.outer(self.x[tau-1, self.arms[tau-1]], self.x[tau-1, self.arms[tau-1]])
                    # self.A += np.dot(self.x[tau-1, self.arms[tau-1]], self.x[tau-1, self.arms[tau-1]])

                self.b[j] += self.reward[j, tau-1, self.arms[tau-1]] * self.x[tau-1, self.arms[tau-1]]
            if not j:
                self.A_inv = np.linalg.inv(self.A)
            self.Theta_hat[j] = np.dot(self.A_inv, self.b[j])

        self.t += 1
        self.Psi = np.append(self.Psi, self.t)

        # self.A[chosen_arm] += np.outer(chosen_arm_features, chosen_arm_features)
        # self.b[chosen_arm] += reward * chosen_arm_features
