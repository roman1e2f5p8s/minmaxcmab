'''
Inspired from:
https://www.kaggle.com/phamvanvung/cb-linucb/notebook
https://github.com/etiennekintzler/visualize_bandit_algorithms/blob/master/linUCB.ipynb
'''

import numpy as np
from linucb import LinUCB
from alg import Alg
import dummy_data
import matplotlib.pyplot as plt


np.random.seed(123)

FEATURE_DIM = 5
N_ARMS = 16
N_TRIALS = 10000
T = N_TRIALS
ALPHA = 2.5

BEST_ARMS = [3, 7, 9, 15]

LINUCB = (1 == 1)
ALG = (1 == 1)
TEST = (1 == 0)


def true_theta(n_arms, feature_dim):
    theta = np.random.normal(size=(n_arms, feature_dim), scale=0.25)
    theta[BEST_ARMS] = theta[BEST_ARMS] + 1

    return theta


def reward_func(theta, context):
    signal = np.dot(theta, context)
    noise = 0.009 * np.random.randn()

    return signal + noise


DATA = np.random.uniform(low=0, high=1, size=(N_TRIALS, N_ARMS, FEATURE_DIM))
TRUE_THETA = true_theta(N_ARMS, FEATURE_DIM)

if TEST:
    DATA = dummy_data.DATA
    TRUE_THETA = dummy_data.TRUE_THETA
    NOISE = dummy_data.NOISE
    N_TRIALS, N_ARMS, FEATURE_DIM = DATA.shape
    T = N_TRIALS


EXPECTED_REWARDS = np.array([
        np.max([reward_func(TRUE_THETA[arm], DATA[t, arm]) for arm in range(N_ARMS)]) \
        for t in np.arange(N_TRIALS)
        ])

plt.figure(figsize=(12.5, 7.5))

if LINUCB:
    linucb = LinUCB(feature_dim=FEATURE_DIM, n_arms=N_ARMS, alpha=ALPHA)
    estimated_rewards_l = np.empty(N_TRIALS)

if ALG:
    alg = Alg(FEATURE_DIM, N_ARMS, DATA[0], TRUE_THETA)
    estimated_rewards_a = np.empty(N_TRIALS)
    ub = np.empty((N_TRIALS, N_ARMS))

for t in range(T):
    print('t={}'.format(t), end='\r')
    features = DATA[t]
    
    if LINUCB:
        chosen_arm = linucb.choose_arm(features)
        reward = reward_func(TRUE_THETA[chosen_arm], features[chosen_arm])
        linucb.update(chosen_arm, features[chosen_arm], reward)
        estimated_rewards_l[t] = reward

    if ALG:
        chosen_arm = alg.choose_arm(features)
        reward = reward_func(TRUE_THETA[chosen_arm], features[chosen_arm])
        alg.update(chosen_arm, features[chosen_arm], reward, TRUE_THETA)
        estimated_rewards_a[t] = reward
        ub[t] = alg.err

if ALG:
    theta_err = [np.linalg.norm(alg.Theta[arm] - TRUE_THETA[arm]) / np.linalg.norm(TRUE_THETA[arm]) \
            for arm in range(N_ARMS)]
    print('Theta error:', theta_err)

if LINUCB:
    regret_l = np.cumsum(EXPECTED_REWARDS[:T] - estimated_rewards_l[:T])
    plt.plot(regret_l, label='LinUCB')
if ALG:
    regret_a = np.cumsum(EXPECTED_REWARDS[:T] - estimated_rewards_a[:T])
    plt.plot(regret_a, label='Alg')
    # for arm in range(N_ARMS):
        # plt.plot(ub[:, arm], label=str(arm))

plt.xlabel('Trials')
plt.ylabel('Regret')
plt.legend()
plt.show()
