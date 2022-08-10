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
        self.r = 2
        # print('r = ', self.r)
        # exit()
        
        assert (T > 0 and type(T) == int)
        self.T = T
        self.t = 0

        self.S = int(np.floor(np.log(T)))

        self.Psi = [np.empty(shape=0, dtype=np.int32) for s in range(self.S)]

        self.bmm = BMM(feature_dim=feature_dim, n_arms=n_arms, eps=eps, delta=delta, v=v, T=T, r=self.r)

        self.rewards = np.zeros(shape=self.r, dtype=np.float64)
    
    def choose_arm(self, features):
        '''
            - features: array of shape (n_arms, feature_dim)
        '''
        s = 1
        A_hat = np.arange(1, self.n_arms + 1)
        chosen_arm = None

        while chosen_arm is None:
            r_hat, w = self.bmm.step(rewards=self.rewards, features=features, Psi=self.Psi[s - 1])

            if (w[A_hat - 1] <= 1.0 / np.sqrt(self.T)).all():
                max_val = np.max(r_hat[A_hat - 1] + w[A_hat - 1])
                chosen_arm = np.where((r_hat + w) == max_val)[0][0] # arm found
            elif (w[A_hat - 1] > np.power(2.0, -s)).any(): # choose arm and update Psi
                idx = np.where(w[A_hat - 1] > np.power(2.0, -s))[0][0]
                chosen_arm = np.where(w == w[A_hat - 1][idx])[0][0] # arm found
                self.Psi[s - 1] = np.append(self.Psi[s - 1], self.t)
            else: # (w[A_hat - 1] <= np.power(2, -s)).all(): # update A_hat, s
                A_hat = np.array([a for a in A_hat if r_hat[A_hat - 1] + w[A_hat - 1] >= \
                        np.max(r_hat[A_hat - 1] + w[A_hat - 1]) - np.power(2.0, 1 - s)], dtype=int)
                s += 1
        self.t += 1

        # self.bmm.arms = np.append(self.bmm.arms, choose_arm)
        self.bmm.x = np.append(self.bmm.x, [features[chosen_arm]], axis=0)

        return chosen_arm
