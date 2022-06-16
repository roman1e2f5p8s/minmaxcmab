import pickle
import numpy as np
import matplotlib.pyplot as plt

from linucb import LinUCB
from dlinucbd import DLinUCBD
from alg import Alg
from bmm import Bmm


np.random.seed(123)


FEATURE_DIM = 10
N_ARMS = 20
N_TRIALS = 9000 # number of total time steps
T = N_TRIALS    # number of time steps to be used for learning
assert T <= N_TRIALS


ALPHA = 2.5  # alpha parameter in LinUCB
SIGMA = 1.0  # scale of noise distribution 
STATIONARY = False # setting this to True will make DLinUCB's gamma = 1!!!


BABY_EXAMPLE = False  # set to True to reproduce examples from the DLinUCB paper
THETA_CHANGE_POINT = 2000
SLOW_VAR = False  # set to True to simulate slow varying Theta

BMM_TEST = True # generate the same data as in the BMM paper
BMM_EPS = 1
BMM_DELTA = 0.01
BMM_V = 1


UNIFORM_FEATURES = True   # generate features using uniform distribution, otherwise use normal distr.
SCALE_IN_FEATURES = 0.51
SCALE_IN_THETA = 0.25
BEST_ARMS = [] # was [3, 7, 9, 15]


N_RUNS = 1  # number of realizations
MEAN = True  # plot mean or median result
# for plotting error bars
EVERY_ERR = int(T / 20)
SHIFT_ERR = int(EVERY_ERR / 10)

# set any number to 0 to not run a given algorithm
LINUCB = (1 == 0)
DLINUCB = (1 == 0)
ALG = (1 == 0)
BMM = (1 == 1)


def get_features(n_trials, n_arms, feature_dim, scale):
    ''' Generates features
    '''
    features = np.empty(shape=(n_trials, n_arms, feature_dim))

    for t in range(n_trials):
        if UNIFORM_FEATURES:
            features[t] = np.random.uniform(low=0, high=1, size=(n_arms, feature_dim))
        elif BMM_TEST:
            for a in range(n_arms):
                features[t, a] = np.random.uniform(low=0, high=1, size=feature_dim)
                features[t, a] = features[t, a] / np.linalg.norm(features[t, a])
        else:
            features[t] = np.random.normal(size=(n_arms, feature_dim), scale=scale)

        if DLINUCB:
            try:
                assert np.linalg.norm(features[t], axis=1).all() <= 1
            except AssertionError:
                print("||Features|| = {} for t = {}".format(np.linalg.norm(features[t]), t))
                exit()

    return features


def get_true_theta(n_trials, n_arms, feature_dim, scale, stationary):
    '''Generates true Theta
    '''
    theta = np.empty(shape=(n_trials, n_arms, feature_dim))

    if BABY_EXAMPLE:
        if SLOW_VAR:
            # slowly-varying scenarios
            for t in range(n_trials):
                if t < 3000:
                    dx = 1.0/3000
                    x1 = 1 - dx*t
                    x2 = np.sqrt(1 - x1**2)
                    theta[t, :] = np.array([x1, x2])
                else:
                    theta[t, :] = np.array([0, 1])
                if DLINUCB:
                    try:
                        assert np.linalg.norm(theta[t], axis=1).all() <= 1
                    except AssertionError:
                        print("||Theta|| = {} for t = {}".format(np.linalg.norm(theta[t]), t))
                        exit()
            return theta
        else:
            # abruptly-changing scenarios
            for t in range(n_trials):
                if t < 1000:
                    theta[t, :] = np.array([1, 0])
                elif t >= 1000 and t < 2000:
                    theta[t, :] = np.array([-1, 0])
                elif t >= 2000 and t < 3000:
                    theta[t, :] = np.array([0, 1])
                else:
                    theta[t, :] = np.array([0, -1])
                if DLINUCB:
                    try:
                        assert np.linalg.norm(theta[t], axis=1).all() <= 1
                    except AssertionError:
                        print("||Theta|| = {} for t = {}".format(np.linalg.norm(theta[t]), t))
                        exit()
            return theta
    elif BMM_TEST:
        for t in range(n_trials):
            for a in range(n_arms):
                theta[t, a] = np.ones(shape=feature_dim, dtype=np.float64) / np.sqrt(feature_dim)
        return theta
    
    const_theta = np.random.normal(size=(n_arms, feature_dim), scale=scale)
    new_theta = const_theta

    for t in range(n_trials):
        if stationary:
            theta[t] = const_theta
        else:
            if not ((t + 1) % THETA_CHANGE_POINT):
                new_theta = np.random.normal(size=(n_arms, feature_dim), scale=scale)
            theta[t] = new_theta
        if BEST_ARMS:
            theta[t, BEST_ARMS] = theta[t, BEST_ARMS] + 1
        
        if DLINUCB:
            try:
                assert np.linalg.norm(theta[t], axis=1).all() <= 1
            except AssertionError:
                print("||Theta|| = {} for t = {}".format(np.linalg.norm(theta[t]), t))
                exit()

    return theta


def get_reward(theta, context, noise):
    '''Returns reward
    '''
    signal = np.dot(theta, context)

    return signal + noise


if LINUCB:
    regret_l = np.empty((N_RUNS, N_TRIALS))
if DLINUCB:
    regret_d = np.empty((N_RUNS, N_TRIALS))
if ALG:
    regret_a = np.empty((N_RUNS, N_TRIALS))
if BMM:
    regret_b = np.empty((N_RUNS, N_TRIALS))


# run N_RUNS realizations
for run in range(N_RUNS):
    print('Realization {} out of {}'.format(run+1, N_RUNS))
    DATA = get_features(N_TRIALS, N_ARMS, FEATURE_DIM, SCALE_IN_FEATURES)
    TRUE_THETA = get_true_theta(N_TRIALS, N_ARMS, FEATURE_DIM, SCALE_IN_THETA, STATIONARY)
    # NOISE = SIGMA * np.random.randn(N_TRIALS)
    if BMM_TEST:
        NOISE = np.random.standard_t(df=3, size=N_TRIALS)
    else:
        NOISE = np.random.laplace(scale=SIGMA, size=N_TRIALS)
    
    EXPECTED_REWARDS = np.array([
            np.max([get_reward(TRUE_THETA[t, arm], DATA[t, arm], NOISE[t]) for arm in range(N_ARMS)]) \
            for t in np.arange(N_TRIALS)
            ])
    
    
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
        BT = np.array([sum([np.linalg.norm(TRUE_THETA[t, a] - TRUE_THETA[t+1, a]) \
                for t in range(N_TRIALS-1)]) for a in range(N_ARMS)])
        # print(BT)
        GAMMA = 1 - np.power(BT / (FEATURE_DIM * N_TRIALS) , 2.0/3)
        dlinucb = DLinUCBD(feature_dim=FEATURE_DIM, n_arms=N_ARMS, delta=DELTA, sigma=SIGMA, 
                lambda_=LAMBDA, L=L, S=S, gamma=GAMMA)
        # print(dlinucb)
        estimated_rewards_d = np.empty(N_TRIALS)
    
    if ALG:
        alg = Alg(N_TRIALS, FEATURE_DIM, N_ARMS, DATA[0], TRUE_THETA)
        # ub = np.empty((N_TRIALS, N_ARMS))
        estimated_rewards_a = np.empty(N_TRIALS)
    
    if BMM:
        bmm = Bmm(
                feature_dim=FEATURE_DIM,
                n_arms=N_ARMS,
                eps=BMM_EPS,
                delta=BMM_DELTA,
                v=BMM_V,
                T=N_TRIALS,
                )
        estimated_rewards_b = np.empty(N_TRIALS)

    # run T time steps
    for t in range(T):
        print('t={} of {}'.format(t+1, T), end='\r')
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
            alg.update(chosen_arm, features[chosen_arm], reward, TRUE_THETA) #TODO: change to 3D 
            estimated_rewards_a[t] = reward
    
        if BMM:
            chosen_arm = bmm.choose_arm(features)
            rewards = [get_reward(TRUE_THETA[t, chosen_arm], features[chosen_arm], NOISE[t])]
            rewards = rewards + [get_reward(TRUE_THETA[t, chosen_arm], features[chosen_arm],
                np.random.standard_t(df=3)) for j in range(1, bmm.r)]
            rewards = np.array(rewards)
            bmm.update(rewards)
            estimated_rewards_b[t] = np.median(rewards)
    
    # compute theta error for min-max CMAB algorithm
    # if ALG:
        # theta_err = [np.linalg.norm(alg.Theta[arm] - TRUE_THETA[arm]) /\ 
                # np.linalg.norm(TRUE_THETA[arm]) for arm in range(N_ARMS)]
        # print('Theta error:', theta_err)

    if LINUCB:
        regret_l[run] = np.cumsum(EXPECTED_REWARDS[:T] - estimated_rewards_l[:T])
    if DLINUCB:
        regret_d[run] = np.cumsum(EXPECTED_REWARDS[:T] - estimated_rewards_d[:T])
    if ALG:
        regret_a[run] = np.cumsum(EXPECTED_REWARDS[:T] - estimated_rewards_a[:T])
    if BMM:
        regret_b[run] = np.cumsum(EXPECTED_REWARDS[:T] - estimated_rewards_b[:T])


# plot the results
plt.figure(figsize=(12.5, 7.5))
TRIALS = np.arange(N_TRIALS)[:T]

if LINUCB:
    s = 0
    if MEAN:
        avg = np.mean(regret_l, axis=0)
        std = np.std(regret_l, axis=0)
        low = avg - 2.626 * std / np.sqrt(N_RUNS)
        up = avg + 2.626 * std / np.sqrt(N_RUNS)
    else:
        avg = np.median(regret_l, axis=0)
        low = np.percentile(a=regret_l, q=25, axis=0)
        up = np.percentile(a=regret_l, q=75, axis=0)
    plt.plot(avg, label='LinUCB', color='red')
    plt.errorbar(x=TRIALS[s*SHIFT_ERR:][::EVERY_ERR], y=avg[s*SHIFT_ERR:][::EVERY_ERR],
            yerr=[(avg - low)[s*SHIFT_ERR:][::EVERY_ERR], (up - avg)[s*SHIFT_ERR:][::EVERY_ERR]],
            color='red', linestyle='', capsize=3)

if DLINUCB:
    s = 1
    if MEAN:
        avg = np.mean(regret_d, axis=0)
        std = np.std(regret_d, axis=0)
        low = avg - 2.626 * std / np.sqrt(N_RUNS)
        up = avg + 2.626 * std / np.sqrt(N_RUNS)
    else:
        avg = np.median(regret_d, axis=0)
        low = np.percentile(a=regret_d, q=25, axis=0)
        up = np.percentile(a=regret_d, q=75, axis=0)
    # pickle.dump(avg, open('normal_05.pkl', 'wb'))
    plt.plot(avg, label='D-LinUCB', color='blue')
    plt.errorbar(x=TRIALS[s*SHIFT_ERR:][::EVERY_ERR], y=avg[s*SHIFT_ERR:][::EVERY_ERR],
            yerr=[(avg - low)[s*SHIFT_ERR:][::EVERY_ERR], (up - avg)[s*SHIFT_ERR:][::EVERY_ERR]],
            color='blue', linestyle='', capsize=3)

if ALG:
    s = 2
    if MEAN:
        avg = np.mean(regret_a, axis=0)
        std = np.std(regret_a, axis=0)
        low = avg - 2.626 * std / np.sqrt(N_RUNS)
        up = avg + 2.626 * std / np.sqrt(N_RUNS)
    else:
        avg = np.median(regret_a, axis=0)
        low = np.percentile(a=regret_a, q=25, axis=0)
        up = np.percentile(a=regret_a, q=75, axis=0)
    plt.plot(avg, label='Min-Max CMAB', color='green')
    plt.errorbar(x=TRIALS[s*SHIFT_ERR:][::EVERY_ERR], y=avg[s*SHIFT_ERR:][::EVERY_ERR],
            yerr=[(avg - low)[s*SHIFT_ERR:][::EVERY_ERR], (up - avg)[s*SHIFT_ERR:][::EVERY_ERR]],
            color='green', linestyle='', capsize=3)

if BMM:
    s = 3
    if MEAN:
        avg = np.mean(regret_b, axis=0)
        std = np.std(regret_b, axis=0)
        low = avg - 2.626 * std / np.sqrt(N_RUNS)
        up = avg + 2.626 * std / np.sqrt(N_RUNS)
    else:
        avg = np.median(regret_b, axis=0)
        low = np.percentile(a=regret_b, q=25, axis=0)
        up = np.percentile(a=regret_b, q=75, axis=0)
    plt.plot(avg, label='BMM', color='orange')
    plt.errorbar(x=TRIALS[s*SHIFT_ERR:][::EVERY_ERR], y=avg[s*SHIFT_ERR:][::EVERY_ERR],
            yerr=[(avg - low)[s*SHIFT_ERR:][::EVERY_ERR], (up - avg)[s*SHIFT_ERR:][::EVERY_ERR]],
            color='red', linestyle='', capsize=3)

if BABY_EXAMPLE:
    if SLOW_VAR:
        plt.title('Synthetic data in slowly-varying scenarios')
    else:
        plt.title('Synthetic data in abruptly-changing scenarios')

plt.xlabel('Trials')
plt.ylabel('Regret')
plt.legend()
plt.show()
