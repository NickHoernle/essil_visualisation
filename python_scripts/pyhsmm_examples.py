import sys
import essil_prepocessing as pre
import numpy as np
import matplotlib.pyplot as plt
import pyhsmm
import pyhsmm.basic.distributions as distributions
import pickle
from scipy.special import logsumexp

BIOMES = pre.BIOMES
folder = 'hdp_test8'
# df = pre.load_essil_file(f'../essil_data/hdp_test10/student_trial_2019-03-12_12-10-43.csv')
df = pre.load_essil_file(f'../essil_data/hdp_test8/student_trial_2019-03-12_11-44-30.csv')
d = df.drop_duplicates(subset='_video_seconds')
data = (d[BIOMES].diff().fillna(0).values / d.seconds.diff().fillna(1).values[:, None]).T
rain_mask = d[[b.replace('Water', 'Raining') for b in BIOMES]].values.astype(int).T == 1
idx = np.where(~rain_mask, np.arange(rain_mask.shape[1]), 0)
np.maximum.accumulate(idx, axis=1, out=idx)
data[rain_mask] = data[np.nonzero(rain_mask)[0], idx[rain_mask]]
data = data.T
data /= np.percentile(data, 99.0)

df2 = pre.load_essil_file(f'../essil_data/hdp_test10/student_trial_2019-03-12_12-10-43.csv')
d2 = df2.drop_duplicates(subset='_video_seconds')
data2 = (d2[BIOMES].diff().fillna(0).values / d2.seconds.diff().fillna(1).values[:, None]).T
rain_mask = d2[[b.replace('Water', 'Raining') for b in BIOMES]].values.astype(int).T == 1
idx = np.where(~rain_mask, np.arange(rain_mask.shape[1]), 0)
np.maximum.accumulate(idx, axis=1, out=idx)
data2[rain_mask] = data2[np.nonzero(rain_mask)[0], idx[rain_mask]]
data2 = data2.T
data2 /= np.percentile(data2, 99.0)

def get_mode(array):
    return max(set(array), key=array.count)

def update_transmatrix(t_mx, states):
    trans_matrix = np.zeros(shape=(len(states), len(states)))
    for i,s1 in enumerate(states):
        for j,s2 in enumerate(states):
            trans_matrix[i,j] = t_mx[s1,s2]
    return trans_matrix

def get_DIC(hmm, models, data, data2):
    L = hmm.log_likelihood(data) + hmm.log_likelihood(data2)
    llSum = np.sum([m.log_likelihood(data) for m in models]) + np.sum([m.log_likelihood(data2) for m in models])
    P = 2 * (L - (1 / len(models) * llSum))
    DIC = -2 * (L - P)
    return DIC

def get_WAIC(hmm, models, data, data2):

    # log ( mean ( likelihood for each sample ) )
    LPPD = logsumexp([m.log_likelihood(data) for m in models]) + logsumexp([m.log_likelihood(data2) for m in models]) - np.log(len(models))

    ll = [m.log_likelihood(data) for m in models]
    ll2 = [m.log_likelihood(data2) for m in models]

    P = logsumexp(ll) - np.log(len(models)) - np.mean(ll)
    P2 = logsumexp(ll2) - np.log(len(models)) - np.mean(ll2)

    WAIC = -2 * (LPPD - 2*(P + P2))
    return WAIC


scores_L = []
scores_WAIC = []
scores_DIC = []

for i in range(15):
    L = []
    DIC = []
    WAIC = []

    for k,m in enumerate([
        'model_1.pkl',
        'model_5.pkl',
        'model_10.pkl',
        'model_50.pkl',
        'model_100.pkl',
        'model_150.pkl',
        'model_200.pkl',
        'model_300.pkl',
        'model_500.pkl',
        'model_700.pkl',
        'model_geom.pkl'
    ]):
        with open('/Users/nickhoernle/edinburgh/essil_repository/regime_test_visualisation/data/2019-08-21-hdp8/models/'+m, 'rb') as f:
            model = pickle.load(f)

        z_init = np.zeros(shape=(len(data2),))
        z_init[:((len(data2) // 5) * 5)] = np.random.randint(0, len(model.obs_distns), size=(len(data2) // 5)).repeat(5)
        model.add_data(data2, stateseq=z_init)

        [model.resample_and_copy() for p in range(100)]

        models = [model.resample_and_copy() for p in range(75)]

        states = np.concatenate([model.stateseqs[0] for model in models] + [model.stateseqs[1] for model in models])

        # states = [get_mode(list(vec)) for vec in states.T]
        best_states = np.unique(states)
        # best_states = np.arange(0, len(models[0].obs_distns))

        dist_obs = []
        for state in best_states:

            mdls = [model for model in models]
            # weights = np.mean(np.log([model.obs_distns[state].params['weights']['weights'] for model in mdls]), axis=0)
            # mu1 = np.mean([model.obs_distns[state].params['components'][0]['mu']
            #                    for model in mdls], axis=0)
            # mu2 = np.mean([model.obs_distns[state].params['components'][1]['mu']
            #                    for model in mdls], axis=0)
            # sig1 = np.mean([model.obs_distns[state].params['components'][0]['sigma']
            #                    for model in mdls], axis=0)
            # sig2 = np.mean([model.obs_distns[state].params['components'][1]['sigma']
            #                    for model in mdls], axis=0)
            #
            # params1 = {'mu': mu1, 'sigma': sig1}
            # params2 = {'mu': mu2, 'sigma': sig2}
            # weights -= logsumexp(weights)
            mu1 = np.mean([model.obs_distns[state].params['mu'] for model in mdls], axis=0)
            sig1 = np.mean([model.obs_distns[state].params['sigma'] for model in mdls], axis=0)
            params1 = {'mu': mu1, 'sigma': sig1}
            dist_obs.append(distributions.GaussianFixed(**params1))
            # dist_obs.append(distributions.MixtureDistribution(
            #     alpha_0=np.exp(weights),
            #     weights=np.exp(weights),
            #     components=[distributions.GaussianFixed(**params1),
            #                 distributions.GaussianFixed(**params2)]))

        # trans_matrix = np.mean(np.log([model.trans_distn.trans_matrix for model in models]), axis=0)
        trans_matrix = np.mean(np.log([update_transmatrix(model.trans_distn.trans_matrix, best_states) for model in models]), axis=0)
        trans_matrix -= logsumexp(trans_matrix, axis=1)[:, None]

        hmm = pyhsmm.models.HMM(obs_distns=dist_obs,
                                trans_matrix=np.exp(trans_matrix))

        # use log space
        L.append(hmm.log_likelihood(data) + hmm.log_likelihood(data2))
        DIC.append(get_DIC(hmm, models, data, data2))
        WAIC.append(get_WAIC(hmm, models, data, data2))

    scores_L.append(L)
    scores_WAIC.append(WAIC)
    scores_DIC.append(DIC)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}
import matplotlib
matplotlib.rc('font', **font)
fig = plt.figure(figsize=(10,7))

# for score, label in zip([scores_L, scores_DIC, scores_WAIC], ['Log Likelihood', 'DIC', 'WAIC']):
ax1, ax2 = plt.gca(), plt.twinx()
for score, label, ax, c in zip([scores_DIC, scores_WAIC], ['DIC', 'WAIC'], [ax1, ax2], ['blue', 'orange']):

    with open(f'results_{label}.pkl', 'wb') as f:
        pickle.dump(score, f)
    K = [1, 5, 10, 50, 100, 150, 200, 300, 500, 700]
    mean = np.mean(score, axis=0)
    std = np.std(score, axis=0)

    ax.errorbar(np.arange(len(mean)), mean, yerr=std, lw=0, marker='.', ms=20, elinewidth=1, label=label, c=c)

ax1.set_xticks(np.arange(len(mean)))
ax1.set_xticklabels(["$MK_{"+str(k)+"}$" for k in K]+['FB'],  rotation=80)
# ax2.set_xticks([])
# ax2.set_xticklabels([])
# ax2.xticks([], [])
# ax = plt.gca()
plt.title("DIC and WAIC scores as a function of model")
# plt.legend(loc='best')
ax1.set_ylabel('Information Criteria')
plt.xlabel('Model')
plt.show()
