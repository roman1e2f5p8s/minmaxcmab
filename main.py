'''
Inspired from:
https://www.kaggle.com/phamvanvung/cb-linucb/notebook
https://github.com/etiennekintzler/visualize_bandit_algorithms/blob/master/linUCB.ipynb
'''

import numpy as np
from linucb import LinUCB
import matplotlib.pyplot as plt


np.random.seed(123)

FEATURE_DIM = 5
N_ARMS = 10
ALPHA = 2.5
N_TRIALS = 5000
# BEST_ARMS = [3, 7, 9, 15]


DATA = np.random.uniform(low=0, high=1, size=(N_TRIALS, N_ARMS, FEATURE_DIM))


def true_theta(n_arms, feature_dim):
    theta = np.random.normal(size=(n_arms, feature_dim), scale=0.25)
    # theta[BEST_ARMS] = theta[BEST_ARMS] + 1

    return theta


def reward_func(arm, context, theta, scale_noise=0.1):
    signal = np.dot(theta[arm], context)
    noise = np.random.normal(scale=scale_noise)

    return signal + noise


TRUE_THETA = true_theta(N_ARMS, FEATURE_DIM)
EXPECTED_REWARDS = np.array([
        np.max([reward_func(arm, DATA[t, arm], TRUE_THETA) for arm in range(N_ARMS)]) \
        for t in np.arange(N_TRIALS)
        ])
estimated_rewards = np.empty(N_TRIALS)

linucb = LinUCB(feature_dim=FEATURE_DIM, n_arms=N_ARMS, alpha=ALPHA)

for t in range(N_TRIALS):
    print('t={}'.format(t), end='\r')
    features = DATA[t]
    chosen_arm = linucb.choose_arm(features)

    reward = reward_func(chosen_arm, features[chosen_arm], TRUE_THETA)
    linucb.update(chosen_arm, features[chosen_arm], reward)
    estimated_rewards[t] = reward

regret = np.cumsum(EXPECTED_REWARDS - estimated_rewards)

plt.plot(regret)
plt.xlabel('Trials')
plt.ylabel('Regret')
# plt.ylim([0,5000])
plt.show()
