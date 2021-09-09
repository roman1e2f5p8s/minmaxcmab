'''
Inspired from:
https://www.kaggle.com/phamvanvung/cb-linucb/notebook
https://github.com/etiennekintzler/visualize_bandit_algorithms/blob/master/linUCB.ipynb
'''

import numpy as np
from linucb import LinUCB
from alg import Alg
import test
import matplotlib.pyplot as plt


np.random.seed(123)

FEATURE_DIM = 5
N_ARMS = 16
N_TRIALS = 2000
ALPHA = 2.5
LINUCB = (1 == 1)
ALG = (1 == 1)
TEST = (1 == 0)
BEST_ARMS = [3, 7, 9, 15]


def true_theta(n_arms, feature_dim):
    theta = np.random.normal(size=(n_arms, feature_dim), scale=0.25)
    theta[BEST_ARMS] = theta[BEST_ARMS] + 1

    return theta


def reward_func(arm, context, theta, scale_noise=0.1):
    signal = np.dot(theta[arm], context)
    noise = np.random.normal(scale=scale_noise)# * (0.5/N_TRIALS)
    # noise = 0.0

    return signal + noise


DATA = np.random.uniform(low=0, high=1, size=(N_TRIALS, N_ARMS, FEATURE_DIM))
TRUE_THETA = true_theta(N_ARMS, FEATURE_DIM)

if TEST:
    DATA = test.DATA
    TRUE_THETA = test.TRUE_THETA
    N_TRIALS, N_ARMS, FEATURE_DIM = DATA.shape


EXPECTED_REWARDS = np.array([
        np.max([reward_func(arm, DATA[t, arm], TRUE_THETA) for arm in range(N_ARMS)]) \
        for t in np.arange(N_TRIALS)
        ])

plt.figure(figsize=(12.5, 7.5))

if LINUCB:
    linucb = LinUCB(feature_dim=FEATURE_DIM, n_arms=N_ARMS, alpha=ALPHA)
    estimated_rewards_l = np.empty(N_TRIALS)

if ALG:
    alg = Alg(feature_dim=FEATURE_DIM, n_arms=N_ARMS, alpha=ALPHA)
    estimated_rewards_a = np.empty(N_TRIALS)

one_minus_sum = 1
sum_x = 0
etas = np.empty(N_TRIALS)
alphas = np.empty((N_ARMS, N_TRIALS))

for t in range(N_TRIALS):
    print('t={}'.format(t), end='\r')
    features = DATA[t]
    
    if LINUCB:
        chosen_arm = linucb.choose_arm(features)
        reward = reward_func(chosen_arm, features[chosen_arm], TRUE_THETA)
        linucb.update(chosen_arm, features[chosen_arm], reward)
        estimated_rewards_l[t] = reward

    if ALG:
        chosen_arm = alg.choose_arm(features)
        alphas[:, t] = alg.alpha

        # dot = alg.R.dot(features[chosen_arm]).dot(features[chosen_arm])
        # eta = np.sqrt(one_minus_sum / dot / N_TRIALS)
        # etas[t] = eta
        # x = one_minus_sum / N_TRIALS # dot * eta**2
        # one_minus_sum -= x
        # sum_x += x
        # assert one_minus_sum >= 0
        # print(eta, x)
        # exit()
    
        reward = reward_func(chosen_arm, features[chosen_arm], TRUE_THETA)# + eta
        alg.update(chosen_arm, features[chosen_arm], reward)
        estimated_rewards_a[t] = reward

# print('sum_x =', sum_x)
# for arm in range(N_ARMS):
    # plt.plot(alphas[arm], label=str(arm))
# plt.legend()
# plt.show()
# exit()

if LINUCB:
    regret_l = np.cumsum(EXPECTED_REWARDS - estimated_rewards_l)
    plt.plot(regret_l, label='LinUCB')
if ALG:
    regret_a = np.cumsum(EXPECTED_REWARDS - estimated_rewards_a)
    plt.plot(regret_a, label='Alg')

plt.xlabel('Trials')
plt.ylabel('Regret')
plt.legend()
plt.show()
