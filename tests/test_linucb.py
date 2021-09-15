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
N_ARMS = 16
ALPHA = 0.0
N_TRIALS = 2000
BEST_ARMS = [3, 7, 9, 15]

# DATA = np.random.uniform(low=0, high=1, size=(N_TRIALS, N_ARMS, FEATURE_DIM))
DATA = np.array([[np.random.uniform(low=0, high=1, size=FEATURE_DIM) for _ in range(N_ARMS)] \
        for _ in range(N_TRIALS)])


def true_theta(n_arms, feature_dim):
    theta = np.array([np.random.normal(size=feature_dim, scale=1/4) for _ in range(n_arms)])
    theta[BEST_ARMS] = theta[BEST_ARMS] + 1

    return theta


def reward_func(arm, context, theta, scale_noise=1/10):
    signal = np.dot(theta[arm], context)
    noise = np.random.normal(scale=scale_noise)

    return signal + noise


TRUE_THETA = true_theta(N_ARMS, FEATURE_DIM)

DATA = np.array([[np.random.uniform(low=0, high=1, size=FEATURE_DIM) for _ in range(N_ARMS)] \
        for _ in range(N_TRIALS)])
TRUE_THETA = true_theta(N_ARMS, FEATURE_DIM)

ave_reward = np.mean([[reward_func(arm=arm, context=DATA[t, arm], theta=TRUE_THETA) 
    for arm in np.arange(N_ARMS)] for t in np.arange(N_TRIALS)], axis=0)

EXPECTED_REWARDS = np.array([
        np.max([reward_func(arm, DATA[t, arm], TRUE_THETA) for arm in range(N_ARMS)]) \
        for t in np.arange(N_TRIALS)
        ])

estimated_rewards = np.empty(N_TRIALS)

payoff_random = np.array([reward_func(arm=np.random.choice(N_ARMS), 
    context=DATA[t, np.random.choice(N_ARMS)], theta=TRUE_THETA) for t in np.arange(DATA.shape[0])])

alpha_to_test = [0, 1, 2.5, 5, 10, 20]
alpha_to_test = [2.5]
plt.figure(figsize=(12.5, 7.5))

for alpha in alpha_to_test:
    linucb = LinUCB(feature_dim=FEATURE_DIM, n_arms=N_ARMS, alpha=alpha)
    
    for t in range(N_TRIALS):
        print('t={}'.format(t), end='\r')
        features = DATA[t]
    
        chosen_arm = linucb.choose_arm(features)
        reward = reward_func(chosen_arm, features[chosen_arm], TRUE_THETA)
        linucb.update(chosen_arm, features[chosen_arm], reward)
        estimated_rewards[t] = reward
    
    regret = np.cumsum(EXPECTED_REWARDS - estimated_rewards)
    
    plt.plot(regret, label='alpha: ' + str(alpha))

# plt.plot(np.cumsum(EXPECTED_REWARDS - payoff_random), label = "random", linestyle='--')
plt.xlabel('Trials')
plt.ylabel('Regret')
plt.legend()
plt.show()
