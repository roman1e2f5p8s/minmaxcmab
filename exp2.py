'''
Inspired from:
https://www.kaggle.com/phamvanvung/cb-linucb/notebook
https://github.com/etiennekintzler/visualize_bandit_algorithms/blob/master/linUCB.ipynb
'''

import numpy as np
from linucb import LinUCB
from dlinucb import DLinUCB
from alg import Alg
# import dummy_data
import matplotlib.pyplot as plt


np.random.seed(123)

FEATURE_DIM = 2
N_ARMS = 1
N_TRIALS = 6000
T = N_TRIALS
ALPHA = 2.5
SIGMA = 0.01

SCALE_IN_FEATURES = 0.01 #0.25

LINUCB = (1 == 1)
DLINUCB = (1 == 1)
ALG = (1 == 0)
TEST = (1 == 0)


def get_features(n_trials, n_arms, feature_dim, scale):
    features = np.empty(shape=(n_trials, n_arms, feature_dim))

    for t in range(n_trials):
        # features[t] = np.random.uniform(low=0, high=1, size=(n_arms, feature_dim))
        features[t] = np.random.normal(size=(n_arms, feature_dim), scale=scale)
        try:
            assert np.linalg.norm(features[t]) <= 1
        except AssertionError:
            print("||Features|| = {} for t = {}".format(np.linalg.norm(features[t]), t))
            exit()

    return features


def get_true_theta(n_trials, n_arms, feature_dim):
    theta = np.empty(shape=(n_trials, n_arms, feature_dim))
    
    for t in range(n_trials):
        if t < 3000:
            theta[t] = np.array([1 - t*1.0/3000, 0 + t*1.0/3000])
        else:
            theta[t] = np.array([0, 1])
        
        try:
            assert np.linalg.norm(theta[t]) <= 1
        except AssertionError:
            print("||Theta|| = {} for t = {}".format(np.linalg.norm(theta[t]), t))
            exit()

    return theta


def get_reward(theta, context, noise):
    signal = np.dot(theta, context)
    # noise = 0.01 * np.random.randn()

    return signal + noise


DATA = get_features(N_TRIALS, N_ARMS, FEATURE_DIM, SCALE_IN_FEATURES)
TRUE_THETA = get_true_theta(N_TRIALS, N_ARMS, FEATURE_DIM)
NOISE = SIGMA * np.random.randn(N_TRIALS)

if TEST:
    DATA = dummy_data.DATA
    TRUE_THETA = dummy_data.TRUE_THETA #TODO: change to 3D TRUE_THETA
    NOISE = dummy_data.NOISE
    N_TRIALS, N_ARMS, FEATURE_DIM = DATA.shape
    T = N_TRIALS


EXPECTED_REWARDS = np.array([
        np.max([get_reward(TRUE_THETA[t, arm], DATA[t, arm], NOISE[t]) for arm in range(N_ARMS)]) \
        for t in np.arange(N_TRIALS)
        ])

plt.figure(figsize=(12.5, 7.5))

if LINUCB:
    linucb = LinUCB(feature_dim=FEATURE_DIM, n_arms=N_ARMS, alpha=ALPHA)
    estimated_rewards_l = np.empty(N_TRIALS)

if DLINUCB:
    DELTA = 0.05
    LAMBDA = 1
    # in the paper, they assume L=S=1 (i.e., rewards are bounded between -1 and 1)
    L = np.ceil(max([np.linalg.norm(DATA[t]) for t in range(N_TRIALS)]))
    S = np.ceil(max([np.linalg.norm(TRUE_THETA[t]) for t in range(N_TRIALS)]))
    # BT is a variation bound (a measure of non-stationarity)
    BT = sum([np.linalg.norm(TRUE_THETA[t] - TRUE_THETA[t+1]) for t in range(N_TRIALS-1)])
    print('BT = {:.4f}'.format(BT))
    GAMMA = 1 - np.power(BT / (FEATURE_DIM * N_TRIALS) , 2.0/3)
    dlinucb = DLinUCB(feature_dim=FEATURE_DIM, n_arms=N_ARMS, delta=DELTA, sigma=SIGMA, 
            lambda_=LAMBDA, L=L, S=S, gamma=GAMMA)
    print(dlinucb)
    estimated_rewards_d = np.empty(N_TRIALS)

if ALG:
    alg = Alg(FEATURE_DIM, N_ARMS, DATA[0], TRUE_THETA) #TODO: change to 3D TRUE_THETA
    estimated_rewards_a = np.empty(N_TRIALS)
    ub = np.empty((N_TRIALS, N_ARMS))

for t in range(T):
    print('t={} of {}'.format(t, N_TRIALS), end='\r')
    features = DATA[t]
    
    if LINUCB:
        chosen_arm = linucb.choose_arm(features)
        reward = get_reward(TRUE_THETA[t, chosen_arm], features[chosen_arm], NOISE[t])
        linucb.update(chosen_arm, features[chosen_arm], reward)
        estimated_rewards_l[t] = reward

    if DLINUCB:
        chosen_arm = dlinucb.choose_arm(t, features)
        reward = get_reward(TRUE_THETA[t, chosen_arm], features[chosen_arm], NOISE[t])
        dlinucb.update(chosen_arm, features[chosen_arm], reward)
        estimated_rewards_d[t] = reward

    if ALG:
        chosen_arm = alg.choose_arm(features)
        reward = get_reward(TRUE_THETA[t, chosen_arm], features[chosen_arm], NOISE[t])
        alg.update(chosen_arm, features[chosen_arm], reward, TRUE_THETA) #TODO: change to 3D TRUE_THETA
        estimated_rewards_a[t] = reward

if ALG:
    theta_err = [np.linalg.norm(alg.Theta[arm] - TRUE_THETA[arm]) / np.linalg.norm(TRUE_THETA[arm]) \
            for arm in range(N_ARMS)] #TODO: change to 3D TRUE_THETA
    print('Theta error:', theta_err)

if LINUCB:
    regret_l = np.cumsum(EXPECTED_REWARDS[:T] - estimated_rewards_l[:T])#/(np.arange(T)+1)
    plt.plot(regret_l, label='LinUCB')
if DLINUCB:
    regret_d = np.cumsum(EXPECTED_REWARDS[:T] - estimated_rewards_d[:T])#/(np.arange(T)+1)
    plt.plot(regret_d, label='D-LinUCB')
if ALG:
    regret_a = np.cumsum(EXPECTED_REWARDS[:T] - estimated_rewards_a[:T])#/(np.arange(T)+1)
    plt.plot(regret_a, label='Alg')

plt.xlabel('Trials')
plt.ylabel('Regret')
plt.legend()
plt.show()
