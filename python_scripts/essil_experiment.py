'''
Author: Nicholas Hoernle
Date: 29 April 2019
Purpose: Control the scripts to generate data for the ESSIL experiment.
'''

import sys
sys.path.insert(0, './..')

import essil_prepocessing as pre
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import experiment_controller as expctrl
import time
import os
import pyhsmm
import pyhsmm.basic.distributions as distributions
import copy
import json
from scipy.signal import convolve
import pickle
from scipy.special import logsumexp

# settings
cw_data2 = '/Users/nickhoernle/edinburgh/essil_repository/essil_data/hdp_test10/student_trial_2019-03-12_12-10-43.csv'
# mov_data = '/Users/nickhoernle/edinburgh/essil_repository/essil_data/hdp_test10/student_trial_2019-03-12_12-10-43.mov'
out_folder = '/Users/nickhoernle/edinburgh/essil_repository/regime_test_visualisation/data/2019-08-21-hdp8-fsim'
# cw_data = '/Users/nickhoernle/edinburgh/essil_repository/essil_data/hdp_test11/log-0-ESSIL_October_Test.csv'
# mov_data = '/Users/nickhoernle/edinburgh/essil_repository/essil_data/hdp_test11/0-ESSIL_October_Test-09-24-50-839.mov'
# out_folder = '/Users/nickhoernle/edinburgh/essil_repository/regime_test_visualisation/data/test11'
cw_data = '/Users/nickhoernle/edinburgh/essil_repository/essil_data/hdp_test8/student_trial_2019-03-12_11-44-30.csv'
mov_data = '/Users/nickhoernle/edinburgh/essil_repository/essil_data/hdp_test8/student_trial_2019-03-12_11-44-30.mov'

BIOMES = pre.BIOMES
n_iter = 2000
# n_iter = 25
n_samps = 250
# n_samps = 25
K = [1,5,10,50,100,150,200,300,500,700]

write_images = True
random_baseline, geometric_full_posterior, poisson_full_posterior = 1, 1, 1
exp_id = 0

def find_or_create_folder(direct):
    if not os.path.exists(direct):
        os.makedirs(direct)
    return direct


def plot_water_levels(df, path_to_save):

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    df[BIOMES].plot(ax=ax, alpha=.9)

    formatter = mpl.ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms)))

    ax.set_xlim([0, 8 * 60])
    ax.set_xticks(np.arange(start=0, stop=520, step=60))
    ax.set_xticklabels(np.arange(start=0, stop=520, step=60))
    ax.xaxis.set_major_formatter(formatter)

    ax.set_title('Water Levels in Different Parts of the Simulation (note the twin axis)')
    ax.set_ylabel('Water amount (biomes)')

    ax.set_xlabel('time')
    plt.savefig(path_to_save)


def get_uniform_selection(k, n):
    full_indexes = np.arange(k)
    return [full_indexes[0]] + [full_indexes[round(k*i/(n))] for i in range(1,n-1)] + [full_indexes[-1]]


def build_obs_hyperparams():
    mu_opts1 = [-.1,  0, .1]
    mu_opts2 = [-.25, 0, .25]
    mu_opts3 = [-.5,  0, .5]

    sd_opts1 = [.1 ** 2,   .05 ** 2,  .1 ** 2]
    sd_opts2 = [.25 ** 2,  .1  ** 2,  .25 ** 2]
    sd_opts3 = [1. ** 2,   .25 ** 2,  1. ** 2]

    mu_opts = [mu_opts1, mu_opts2, mu_opts3]
    sd_opts = [sd_opts1, sd_opts2, sd_opts3]

    obs_dim = 4
    obs_hypparams = []

    for i in range(3):  # desert
        for j in range(3):  # jungle
            for k in range(3):  # plains
                for l in range(3):  # wetlands
                    op = []
                    for o in range(3):
                        op.append(
                            {'mu': np.array([mu_opts[o][i], mu_opts[o][j], mu_opts[o][k], mu_opts[o][l]]),
                             'sigma': np.array([sd_opts[o][i], sd_opts[o][j], sd_opts[o][k], sd_opts[o][l]])
                                        * np.eye(obs_dim)})
                    obs_hypparams.append(op)
    return np.array(obs_hypparams)


def build_obs_hyperparams2(data):
    obs_dim = 4

    return [[
        # {'mu_0': np.zeros(obs_dim),
        #  'sigma_0': np.eye(obs_dim),
        #  'kappa_0': 1e-1,
        #  'nu_0': 1e2},
        # {'mu_0': np.zeros(obs_dim),
        #  'sigma_0': 1.25 * np.eye(obs_dim),
        #  'kappa_0': 1e-1,
        #  'nu_0': 1e1}
        {'mu_0': np.zeros(obs_dim),
         'sigma_0': 1*np.eye(obs_dim),
         'kappa_0': 1,
         'nu_0': 1e1},
        {'mu_0': np.zeros(obs_dim),
         'sigma_0': 1.25 * np.eye(obs_dim),
         'kappa_0': 1,
         'nu_0': obs_dim+2}
    ] for _ in range(25)]


def update_transmatrix(t_mx, states):
    trans_matrix = np.zeros(shape=(len(states), len(states)))
    for i,s1 in enumerate(states):
        for j,s2 in enumerate(states):
            trans_matrix[i,j] = t_mx[s1,s2]
    return trans_matrix


def generate_full_posterior(data, obs_hypparams, name, outfolder=out_folder, data2=[]):
    obs_distns = []
    for pars in obs_hypparams:
        obs_distns.append(distributions.Gaussian(**pars[0]))
        # obs_distns.append(distributions.MixtureDistribution(
        #     weights_obj=distributions.Categorical(alphav_0=np.array([99, 1]), K=2),
        #     # weights_obj=distributions.Categorical(alpha_0=1, K=2),
        #     components=[distributions.Gaussian(**pars[0]) for d in range(2)]))
            #             distributions.Gaussian(**pars[0])]))

    model = pyhsmm.models.WeakLimitStickyHDPHMM(
        # alpha=1e1,
        # gamma=1e1,
        # alpha_a_0=1, alpha_b_0=1/4,
        gamma_a_0=1.,
        gamma_b_0=1/4.,
        alpha_kappa_a_0=1.,
        alpha_kappa_b_0=1/4.,
        rho_c_0=1.,
        rho_d_0=1.,
        init_state_concentration=1e3,
        # dur_distns=dur_distns,
        obs_distns=obs_distns)

    z_init = np.zeros(shape=(len(data),))
    z_init[:((len(data) // 5)*5)] = np.random.randint(0, len(obs_distns), size=(len(data) // 5)).repeat(5)
    model.add_data(data, stateseq=z_init)

    if len(data2) > 0:
        z_init = np.zeros(shape=(len(data2),))
        z_init[:((len(data2) // 5) * 5)] = np.random.randint(0, len(obs_distns), size=(len(data2) // 5)).repeat(5)
        model.add_data(data2, stateseq=z_init)

    return burn_in_model_and_get_hmm(model, name, outfolder)

def run_pyshmm_inference(k, data, obs_hypparams, data2):

    obs_distns = []
    for pars in obs_hypparams:
        obs_distns.append(distributions.Gaussian(**pars[0]))
        # obs_distns.append(distributions.MixtureDistribution(
        #     weights_obj=distributions.Categorical(alphav_0=np.array([99, 1]), K=2),
        #     # weights_obj=distributions.Categorical(alpha_0=1, K=2),
        #     components=[distributions.Gaussian(**pars[0]) for d in range(2)]))
        # #             distributions.Gaussian(**pars[0])]))

    model = pyhsmm.models.WeakLimitStickyHDPHMM(
                kappa=k,
                # alpha=1e2,
                # gamma=1e2,
                alpha_a_0=1., alpha_b_0=1/4.,
                gamma_a_0=1., gamma_b_0=1/4.,
                init_state_concentration=1e3,
                obs_distns=obs_distns)

    # model = pyhsmm.models.WeakLimitStickyHDPHMM(
    #     # alpha=1e1,
    #     # gamma=1e1,
    #     # alpha_a_0=1, alpha_b_0=1 / 4,
    #     gamma_a_0=1, gamma_b_0=1 / 4,
    #     alpha_kappa_a_0=1, alpha_kappa_b_0=1 / 4,
    #     rho_c_0=1, rho_d_0=1 / 4,
    #     init_state_concentration=1e2,
    #     # dur_distns=dur_distns,
    #     obs_distns=obs_distns)

    z_init = np.zeros(shape=(len(data),))
    z_init[:((len(data) // 5)*5)] = np.random.randint(0, len(obs_distns), size=(len(data) // 5)).repeat(5)
    model.add_data(data, stateseq=z_init)

    z_init = np.zeros(shape=(len(data2),))
    z_init[:((len(data2) // 5) * 5)] = np.random.randint(0, len(obs_distns), size=(len(data2) // 5)).repeat(5)
    model.add_data(data2, stateseq=z_init)

    return burn_in_model_and_get_hmm(model, str(int(k)))

def burn_in_model_and_get_hmm(model, name, outfolder=out_folder):

    target = outfolder+f'/models/model_{name}.pkl'
    if (os.path.exists(target) and os.path.getsize(target) > 0):
        with open(target, 'rb') as f:
            model = pickle.load(f)

    else:
        for idx in range(n_iter):
            model.resample_model()

        with open(outfolder+f'/models/model_{name}.pkl', 'wb') as f:
            pickle.dump(model, f)

    models = [model.resample_and_copy() for _ in range(n_samps)]
    states = np.array([model.stateseqs[0] for model in models])

    best_states = sorted(np.unique(states))
    # best_states = np.arange(0, len(models[0].obs_distns))

    dist_obs = []
    for state in best_states:

        mdls = [model for model in models]
        if 'weights' in model.obs_distns[0].params:
            weights = np.mean(np.log([model.obs_distns[state].params['weights']['weights'] for model in mdls]), axis=0)
            weights -= logsumexp(weights)

            mu1 = np.mean([model.obs_distns[state].params['components'][0]['mu']
                             for model in mdls], axis=0)
            mu2 = np.mean([model.obs_distns[state].params['components'][1]['mu']
                             for model in mdls], axis=0)
            sig1 = np.mean([model.obs_distns[state].params['components'][0]['sigma']
                              for model in mdls], axis=0)
            sig2 = np.mean([model.obs_distns[state].params['components'][1]['sigma']
                              for model in mdls], axis=0)

            params1 = {'mu': mu1, 'sigma': sig1}
            params2 = {'mu': mu2, 'sigma': sig2}

            dist_obs.append(distributions.MixtureDistribution(
                alpha_0=np.exp(weights),
                components=[distributions.GaussianFixed(**params1),
                            distributions.GaussianFixed(**params2)]))
        else:
            mu1 = np.mean([model.obs_distns[state].params['mu'] for model in mdls], axis=0)
            sig1 = np.mean([model.obs_distns[state].params['sigma'] for model in mdls], axis=0)
            params1 = {'mu': mu1, 'sigma': sig1}
            dist_obs.append(distributions.GaussianFixed(**params1))

    trans_matrix = np.mean(np.log([update_transmatrix(model.trans_distn.trans_matrix, best_states)
                                      for model in models]), axis=0)
    trans_matrix -= logsumexp(trans_matrix, axis=1)[:, None]

    hmm = pyhsmm.models.HMM(obs_distns=dist_obs,
                            trans_matrix=np.exp(trans_matrix))
    return copy.deepcopy(hmm)



def get_means(model, state):
    obs = model.obs_distns[state]
    if 'weights' in obs.params:

        mu1 = obs.params['weights']['weights'][0] * obs.params['components'][0]['mu']
        mu2 = obs.params['weights']['weights'][1] * obs.params['components'][1]['mu']
        return mu1 + mu2
    else:
        return obs.params['mu']

def get_relabeled_state_seq(model, data):

    state_seq = pre.mode_convolution(model.heldout_viterbi(data), 2)
    # state_seq = model.heldout_viterbi(data)
    # state_seq = pre.mode_convolution(model.stateseqs[0], 2)
    start_state = 0
    stateseq = [start_state]
    means ={start_state: get_means(model, state_seq[0])}

    for t in range(1, len(state_seq)):
        if state_seq[t] != state_seq[t-1]:
            if state_seq[t] == state_seq[t+1]:
                start_state += 1
                means[start_state] = get_means(model, state_seq[t])

        stateseq.append(start_state)

    return np.array(stateseq, dtype=int), means, np.array(state_seq, dtype=int)


def validate_regimes(regimes, state_seq):

    valid_regimes = []
    for regime in regimes:
        if regime == 0:
            # don't want to use the starting regime
            continue
        if (np.sum(state_seq == regime)) < 5:
            # condition 1: the regime must be able to support at least 4 images
            continue
        # if np.max([np.sum(state_seq == regime-1), np.sum(state_seq == regime+1)]) < 2:
            # condition 2: want the neighboring regimes to be long enough to be substantial
            # continue
        valid_regimes.append(regime)
    return np.array(valid_regimes)


def flatten_image(img):
    return np.array(img).reshape(1,-1)
def reform_image(img):
    return np.array(img).reshape(405,255,4).astype(np.uint8)

def generate_representative_sample(indexes, regime_name):

    from PIL import Image
    from sklearn.decomposition import PCA
    from os.path import join

    path = out_folder + '/images/'

    data = np.zeros((len(indexes), 405 * 255 * 4))

    img = Image.open(path + 'img5.png')
    base = flatten_image(img)

    for ix, i in enumerate(indexes):
        f = join(path, 'img'+str(i)+'.png')
        img = Image.open(f)
        data[ix] = flatten_image(img)

    data = data.astype(float) - base.astype(float)
    pca = PCA()

    comp = pca.fit_transform(data)

    final = base.astype(np.uint8)[0]
    mean = (pca.mean_).astype(np.uint8)
    final[mean > 50] = mean[mean > 50]
    layer1 = (pca.components_[0] * 255).astype(np.uint8)
    # final[]
    final[layer1 > 0] = layer1[layer1 > 0]

    fig = plt.figure()
    fig.set_size_inches([4, 5])

    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.imshow(reform_image(final))
    plt.savefig(f'{path}/img_representative_{regime_name}.png', bbox_inches='tight', pad_inches=None)

    plt.close()
    plt.cla()
    plt.clf()
    plt.close(fig)

def get_image_weights(indexes):
    from PIL import Image
    from os.path import join

    path = out_folder + '/images/'

    data = np.zeros((len(indexes), 405 * 255 * 4)).astype(float)

    for ix, i in enumerate(indexes):
        f = join(path, 'img' + str(i) + '.png')
        img = Image.open(f)
        data[ix] = flatten_image(img)

    mean = np.mean(data, axis=0).astype(float)
    probs = np.array([np.linalg.norm(d - mean) for d in data])

    scores = sorted([(a,len(probs)-i) for (i,a) in enumerate(np.argsort(probs))], key=lambda x: x[0])
    scores = np.array([s[1] for s in scores])
    scores = scores / np.sum(scores)
    return scores



def generate_experiment_samples(valid_regimes, state_seq, k, cnt, forward, chosen_time_steps, model, orig_state_seq, data):
    '''

    :param valid_regimes:
    :param state_seq:
    :param k:
    :param cnt:
    :param valid_images:
    :return:
    '''
    experiment = []

    for _, chosen in zip(valid_regimes, chosen_time_steps):

        regime = state_seq[chosen]
        regime_set = [regime]

        candidate_idxs = np.where([s in regime_set for s in state_seq])[0]
        while len(candidate_idxs) <= 9:
            regime_set = regime_set + [np.min(regime_set) - 1] + [np.max(regime_set) + 1]
            candidate_idxs = np.where([s in regime_set for s in state_seq])[0]

        candidate_idxs = np.sort(candidate_idxs)
        candidate_idxs = candidate_idxs[candidate_idxs > 0]
        candidate_idxs = candidate_idxs[candidate_idxs < len(data)]
        candidate_idxs = candidate_idxs[2:-2]

        neighbors = [[np.min(regime_set) - 1, np.max(regime_set) + 1][forward]]
        if neighbors[0] < 1:
            neighbors = [np.max(regime_set) + 1]
        if neighbors[0] > np.max(state_seq) - 1:
            neighbors = [np.min(regime_set) - 1]

        models = [model.obs_distns[idx] for idx in np.unique([orig_state_seq[idx_] for idx_ in candidate_idxs])]

        # if (len(candidate_idxs)) < 5:
        #     print("fucked")
        # candidate_images = [candidate_idxs[i] for i in to_select]

        # from the selected regime, sample an intrusion image
        intrusion_idxs = np.where([s in neighbors for s in state_seq])[0]

        for i in range(10):
            if len(intrusion_idxs) <= 9:
                if neighbors[0] < regime_set[0]:
                    neighbors = neighbors + [np.min(neighbors) - 1]
                else:
                    neighbors = neighbors + [np.max(neighbors) + 1]

                if np.min(neighbors) < 0:
                    neighbors = [np.max(regime_set) + 1]
                    forward = 1-forward
                elif np.max(neighbors) > np.max(state_seq):
                    neighbors = [np.min(regime_set) - 1]
                    forward = 1 - forward

                intrusion_idxs = np.where([s in neighbors for s in state_seq])[0]

                if i == 9:
                    print("problem occurred")

        intrusion_idxs = np.sort(intrusion_idxs)
        intrusion_idxs = intrusion_idxs[intrusion_idxs > 0]
        intrusion_idxs = intrusion_idxs[intrusion_idxs < len(data)]
        intrusion_idxs = intrusion_idxs[2:-2]

        # models_int = [model.obs_distns[idx] for idx in np.unique([orig_state_seq[idx_] for idx_ in intrusion_idxs])]

        # if neighbors[0] < regime_set[0]:
        #     intrusion_image = intrusion_idxs[-4:]
        #     candidate_images = np.array(list([candidate_idxs[2]]) + list(candidate_idxs[3:7]))
        # else:
        #     intrusion_image = intrusion_idxs[:4]
        #     candidate_images = np.array(list(candidate_idxs[-7:-3]) + list([candidate_idxs[-3]]))

        # candidate_mu = np.mean([model.mu for model in models], axis=0)
        # intrusion_mu = np.mean([model.mu for model in models_int], axis=0)

        # generate_representative_sample(intrusion_idxs, f"{globals()['exp_id']}_{neighbors[0]}")
        #
        # difference = np.abs(candidate_mu - intrusion_mu)
        # to_keep = (np.argsort(difference)[np.sort(difference) > 0.2])[::-1]
        # difficulty = 0
        # if len(to_keep) == 0:
        #     to_keep = [int(np.argmax(difference))]
        #     difficulty = 1

        # to_select = get_uniform_selection(len(candidate_idxs), 5)
        p = np.hamming(len(candidate_idxs))
        p = p/np.sum(p)
        to_select =np.random.choice(candidate_idxs, size=4, replace=False, p=p)
        # to_select_int = get_uniform_selection(len(intrusion_idxs), 4)
        p = np.hamming(len(intrusion_idxs))
        p = p / np.sum(p)
        to_select_int = [np.random.choice(intrusion_idxs, p=p)]
        print(k, chosen, len(candidate_idxs), len(intrusion_idxs), to_select)

        experiment.append({'candidate': [int(i) for i in to_select],
                           'intrusion': [int(i) for i in to_select_int],
                           'regime_set': [int(i) for i in regime_set],
                           'intrusion_set': [int(i) for i in neighbors],
                           'intrusion_regime': f"{globals()['exp_id']}_{neighbors[0]}",
                           'kappa': k,
                           'experiment': cnt,
                           'experiment_id': globals()['exp_id'],
                           'hint': [0],
                           'forward': forward,
                           'difficulty': 0})

        globals()['exp_id'] += 1
        if globals()['exp_id'] % 25 == 24:
            globals()['exp_id'] += 1

    return experiment, valid_regimes


def generate_random_baseline_samples(chosen_time_steps, avg_time, data, forward, int_time):

    experiment = []

    for cnt, chosen in enumerate(chosen_time_steps):

        length_of_regime = 0
        while length_of_regime < 20:
            length_of_regime = np.random.poisson(avg_time)

        lookback = np.random.poisson(10)
        while lookback < 8:
            lookback = np.random.poisson(10)

        lookfwd = np.random.poisson(10)
        while lookback < 8:
            lookfwd = np.random.poisson(10)


        if 1-forward:
            intrusion_idxs = list(range(chosen+1, chosen+lookfwd))
            candidate_idxs = list(range(chosen-lookback, chosen-1))
        else:
            candidate_idxs = list(range(chosen+1, chosen+lookfwd))
            intrusion_idxs = list(range(chosen-lookback, chosen-1))

        intrusion_image = [np.random.choice(intrusion_idxs)]

        to_select = get_uniform_selection(len(candidate_idxs), 4)
        candidate_images = [candidate_idxs[i] for i in to_select]

        experiment.append({'candidate': [int(i) for i in candidate_images],
                           'intrusion': [int(i) for i in intrusion_image],
                           'intrusion_regime': f"random",
                           'kappa': 11,
                           'experiment': 11,
                           'forward': forward,
                           'experiment_id': 275+cnt+len(chosen_time_steps)*(1-forward)})

    return experiment


def plot_state_inference(state_seq, means, data, title, df, chosen_time_steps, outfolder=out_folder):

    fig,axes = plt.subplots(4,1,figsize=(15,15))
    axes = axes.flatten()

    formatter = mpl.ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms)))

    for i, ts in enumerate(data.T):

        axes[i].plot(ts[df._video_seconds.astype(int)], label=pre.BIOME_NAMES[i])
        means_ = np.array([means[t][i] for t in state_seq])
        axes[i].plot(means_[df._video_seconds.astype(int)])
        axes[i].set_title(pre.BIOME_NAMES[i])
        axes[i].set_ylim([-2,2])

        for x in np.where(np.diff(state_seq[df._video_seconds.astype(int)]) != 0)[0]:
            axes[i].axvline(x+1, alpha=.05, c='b')

        time_mapping = {v:k for k,v in df._video_seconds.astype(int).to_dict().items()}
        for x in chosen_time_steps:
            axes[i].axvline(time_mapping[x], alpha=.25, c='g')

        axes[i].set_xlim([0, 8 * 60])
        axes[i].set_xticks(np.arange(start=0, stop=520, step=60))
        axes[i].set_xticklabels(np.arange(start=0, stop=520, step=60))
        axes[i].xaxis.set_major_formatter(formatter)

    plt.savefig(outfolder + '/plots' + f'/inference_{str(title)}.png')

def select_regimes_from_time(state_seq, chosen_time_steps):
    valid_regimes = []
    for c in chosen_time_steps:
        valid_regimes.append(state_seq[c])
    return valid_regimes

def process_model(model, data, k, outfolder=out_folder):
    state_seq, means, orig_state = get_relabeled_state_seq(model, data)
    state_seq2 = state_seq.copy()

    find_or_create_folder(outfolder + '/plots')

    regimes = np.unique(state_seq)
    boundary_conditions = np.concatenate([[False], convolve(np.diff(state_seq), [1], mode='same').astype(bool)])

    state_seq[boundary_conditions] = -10

    valid_regimes = select_regimes_from_time(state_seq2, chosen_time_steps)
    state_seqs[cnt] = np.array([s in valid_regimes for s in state_seq2], dtype=int)

    plot_state_inference(state_seq2, means, data, k, df, chosen_time_steps)

    num_regimes = np.median([np.sum(c == state_seq) for c in regimes])

    return (valid_regimes, state_seq, num_regimes, state_seq2, orig_state)

if __name__ == '__main__':

    # read csv from file data
    df = pre.load_essil_file( cw_data )
    find_or_create_folder( out_folder + '/plots')
    plot_water_levels(df, out_folder + '/plots/water.png')

    # create the data for inference
    d = df.drop_duplicates(subset='_video_seconds')
    data = (d[BIOMES].diff().fillna(0).values / d.seconds.diff().fillna(1).values[:, None]).T
    rain_mask = d[[b.replace('Water', 'Raining') for b in BIOMES]].values.astype(int).T == 1

    idx = np.where(~rain_mask, np.arange(rain_mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    data[rain_mask] = data[np.nonzero(rain_mask)[0], idx[rain_mask]]
    data = data.T
    data /= np.percentile(data, 99.0)

    # read csv from file data
    df2 = pre.load_essil_file(cw_data)

    # create the data for inference
    d2 = df2.drop_duplicates(subset='_video_seconds')
    data2 = (d2[BIOMES].diff().fillna(0).values / d2.seconds.diff().fillna(1).values[:, None]).T
    rain_mask = d2[[b.replace('Water', 'Raining') for b in BIOMES]].values.astype(int).T == 1

    idx = np.where(~rain_mask, np.arange(rain_mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    data2[rain_mask] = data2[np.nonzero(rain_mask)[0], idx[rain_mask]]
    data2 = data2.T
    data2 /= np.percentile(data2, 99.0)

    chosen_time_steps = []
    T = len(df._video_seconds.unique())
    time_brackets = np.arange(10,T-10,30)
    num_per_bracket = int(np.ceil(12/len(time_brackets)))
    for i in time_brackets:
        chosen_time_steps += np.random.choice(np.arange(max([10, i]), min([i+30, T])), replace=False, size=num_per_bracket).tolist()

    chosen_time_steps = sorted(np.random.choice(chosen_time_steps, size=12, replace=False))
    chosen_time_steps = [27, 59, 113, 156, 184, 209, 238, 264, 282, 312, 371, 375]
    # chosen_time_steps = [35, 54, 95, 121, 151, 174, 211, 251, 305, 331, 355, 377]
    # chosen_time_steps = [35, 71, 101, 130, 187, 196, 226, 272, 289, 305, 334, 355]
    print(chosen_time_steps)
    # drop a line vertically and select the regime that this intersects with. If the regime is no valid, add the
    # neighboring regimes until it becomes valid!
    # while not valid, add regime to chosen regime.

    # generate the observation hyperparams
    obs_hypparams = build_obs_hyperparams2(data)

    # build the inference model
    experiment_details = []

    state_seqs = np.ones(shape=(len(K), data.shape[0]), dtype=int) * -1
    inf_params = []

    for cnt, k in enumerate(K):

        find_or_create_folder(out_folder + '/models')
        # if os.path.isfile(out_folder + f'/models/model_{int(k)}.pkl' ):
        #     continue

        if k >= 500:
            model = run_pyshmm_inference(k+50, data, obs_hypparams, data2)
        else:
            model = run_pyshmm_inference(k, data, obs_hypparams, data2)
        valid_regimes, state_seq, num_regimes, state_seq2, orig = process_model(model, data, k)

        inf_params.append({
            'valid_regimes': valid_regimes,
            'model': model,
            'num_switches': np.sum(np.diff(state_seq) > 0),
            'num_regimes': num_regimes,
            'k': k,
            'state_seq': state_seq,
            'state_seq2': state_seq2,
            'orig_state_seq': orig
        })

    if geometric_full_posterior:
        # generate the negative binomial duration
        # dur_distns = [distributions.GeometricDuration(alpha_0=100, beta_0=2) for _ in range(len(build_obs_hyperparams2(data)))]
        model = generate_full_posterior(data, obs_hypparams, 'geom', out_folder, data2)
        valid_regimes, state_seq, num_regimes, state_seq2, orig = process_model(model, data, 'geom')
        inf_params.append({
            'valid_regimes': valid_regimes,
            'model': model,
            'num_switches': np.sum(np.diff(state_seq) > 0),
            'num_regimes': num_regimes,
            'k': 10000,
            'state_seq': state_seq,
            'state_seq2': state_seq2,
            'orig_state_seq': orig
        })

    # if poisson_full_posterior:
    #     # generate the negative binomial duration
    #     dur_distns = [distributions.PoissonDuration(alpha_0=200, beta_0=.5) for _ in range(len(build_obs_hyperparams2()))]
    #     model = generate_full_posterior(data, obs_hypparams, dur_distns, 'pois')
    #     valid_regimes, state_seq, num_regimes, state_seq2 = process_model(model, data, 'pois')
    #     inf_params.append({
    #         'valid_regimes': valid_regimes,
    #         'model': model,
    #         'num_switches': np.sum(np.diff(state_seq) > 0),
    #         'num_regimes': num_regimes,
    #         'k': 'pois',
    #         'state_seq': state_seq,
    #         'state_seq2': state_seq2
    #     })
    if len(inf_params) == 0:
        with open(out_folder + '/details.pkl', 'rb') as f:
            params_to_store = pickle.load(f)
        inf_params = params_to_store['inf_params']

    else:
        params_to_store = {
            'inf_params': inf_params,
            'data': data,
            'df': df,
            'chosen_time_steps': chosen_time_steps
        }

    for cnt, inf_param in enumerate(inf_params):

        valid_regimes = inf_param['valid_regimes']
        model = inf_param['model']
        num_switches = inf_param['num_switches']
        num_regimes = inf_param['num_regimes']
        k = inf_param['k']
        orig_state_seq = inf_param['orig_state_seq']
        state_seq = inf_param['state_seq2']

        num_valid_regimes = len(valid_regimes)

        for i in range(2):
            details, chosen_regimes = generate_experiment_samples(valid_regimes, state_seq, k, cnt, i, chosen_time_steps, model, orig_state_seq, data)
            experiment_details += details
            median_valid_regime = np.mean([np.sum(c == state_seq) for c in np.unique(state_seq)])
            print(f'kappa: {k}, '
                  f'num valid regimes: {len(valid_regimes)}, '
                  f'num switches: {num_switches}, '
                  f'median len: {median_valid_regime}')

    if random_baseline:
        avg_time = 0
        for c in params_to_store['chosen_time_steps']:
            avg_time += np.median([np.sum(m['state_seq2'][c]==m['state_seq2']) for m in params_to_store['inf_params']])
        avg_time /= len(params_to_store['chosen_time_steps'])

        int_time = np.median([np.min(np.abs(np.array(exp['candidate'])-exp['intrusion'][0])) for exp in experiment_details])
        print(avg_time, int_time)
        # generate the random baseline
        details = generate_random_baseline_samples(chosen_time_steps, avg_time+int_time+2, data, 0, int_time)
        details += generate_random_baseline_samples(chosen_time_steps, avg_time+int_time+2, data, 1, int_time)

        experiment_details += details

    with open(out_folder + '/assignments.json', 'w') as f:
        f.write(json.dumps(experiment_details, indent=4, sort_keys=True))

    with open(out_folder + '/details.pkl', 'wb') as f:
        pickle.dump(params_to_store, f)

    if write_images:
        # ------------------------------
        # now we can generate the images
        find_or_create_folder(out_folder + '/images')
        expctrl.get_file_images(mov_data, out_folder + '/images')
