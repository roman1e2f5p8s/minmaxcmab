import numpy as np

from bmm import BMM


class SupBMM:
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
        self.t = 0

        self.S = int(np.floor(np.log(T)))

        self.Psi = []
        for s in range(self.S):
            self.Psi += [np.empty(shape=0, dtype=np.int32)]

        self.bmm = BMM(feature_dim=feature_dim, n_arms=n_arms, eps=eps, delta=delta, v=v, T=T)
    
    def step(self, rewards, features):
        self.t += 1
        s = 1
        A_hat = np.arange(1, self.n_arms + 1)

        while True:
            r_hat, w = self.bmm.step(t=self.t, rewards=rewards, features=features, Psi=self.Psi[s - 1])

            if (w[A_hat - 1] <= 1.0 / np.sqrt(self.T)).all():
                max_val = np.max(r_hat[A_hat - 1] + w[A_hat - 1])
                chosen_arm = np.where((r_hat + w) == max_val)[0][0]
            elif (w[A_hat - 1] <= np.power(2, -s)).all(): # update A_hat, s
                A_hat = np.array([a for a in A_hat if r_hat[A_hat - 1] + w[A_hat - 1] >= \
                        np.max(r_hat[A_hat - 1] + w[A_hat - 1]) - np.power(2, 1 - s)], dtype=int)
                s += 1
            else: # choose arm and update Psi
                idx = np.where(w[A_hat - 1] > np.power(2, -s))[0][0]
                chosen_arm = np.where(w == w[A_hat - 1][idx])[0][0]
                # TODO
                self.Psi[s - 1] = np.append(self.Psi[s - 1], self.t)

    
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
