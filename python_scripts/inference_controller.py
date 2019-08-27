'''
inference_controller.py
Author: Nicholas Hoernle
Date: March 2019
'''
import copy
import os
from multiprocessing import Pool

import numpy as np
import pyhsmm
import pyhsmm.basic.distributions as distributions
import hdp_scripts as hdp, essil_prepocessing as pre
import classification_evaluation as hdp_eval
import sys

def run_inference_HDPHMM(y, starting_params, id_, **kwargs):

    params = copy.deepcopy(starting_params)

    sampler = hdp.sicky_HDP_HMM(y, params, 'hdphmm', 'hdphmm', 'mulivariate_normal', hmm_type='1-d-gaussian')

    sampler.sample(num_iter=100)

    theta = sampler.get_sample_theta_params()[-1]
    res = sampler.get_sample_assignments()[:,-1]
    return {
        'assignments': pre.mode_convolution2(pre.mode_convolution(res, window=4), theta, window=10),
        'theta': theta,
        'id': id_
    }

def run_inference_semi_HDPHMM(y, starting_params, id_, **kwargs):

    obs_dim = 1
    Nmax = 8

    obs_hypparams = {'mu_0':np.zeros(obs_dim),
                    'sigma_0':np.eye(obs_dim)*kwargs['sigma'],
                    'kappa_0':1e1,
                    'nu_0':obs_dim+1}
    dur_hypparams = {'alpha_0': 60,
                    'beta_0': 1.5}

    obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
    dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

    posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        alpha_a_0=1.,alpha_b_0=1,
        gamma_a_0=1.,gamma_b_0=1, # better to sample over these; see concentration-resampling.py
        init_state_concentration=10., # pretty inconsequential
        obs_distns=obs_distns,
        dur_distns=dur_distns)

    posteriormodel.add_data(y[:,None])

    for idx in range(150):
        posteriormodel.resample_model()

    thetas = [{'A': posteriormodel.obs_distns[s].mu, 'Cov': posteriormodel.obs_distns[s].sigma} for s in range(Nmax)]

    return {
        'assignments': posteriormodel.stateseqs[0],
        'theta': thetas,
        'id': id_
    }

def run_multithread_inference(algorithm, df, data, numprocs=4):

    pool = Pool(numprocs)

    Y = data

    scale = np.max(np.cov(Y.T)[np.arange(len(Y.T)),np.arange(len(Y.T))])
    kwargs = {'sigma': scale}

    params = {'kappa': 1e2, 'alpha': 1, 'gamma': 1, 'L': 8, 'priors': [0,1,1,scale]}
    results = []

    for i,y in enumerate(Y.T):
        results.append(pool.apply_async(algorithm, args=(y, params, i, ), kwds=kwargs))

    result_assignments = {}
    result_params = {}

    for result in results:
        res_ = result.get()
        result_params[res_['id']] = res_['theta']
        result_assignments[res_['id']] = res_['assignments']

    res_combined = []
    theta_combined = []

    for i, biome in enumerate(pre.BIOME_NAMES):

        theta = result_params[i]
        res = result_assignments[i][df._video_seconds.astype(int)]
        res_combined.append(res)
        theta_combined.append(theta)

        # save the relevant regimes for each biomes
        df[biome+'_specific_regime'] = res
        df[biome+'_specific_theta'] = [theta[r]['A'][0] for r in res]

    mapping = {}
    theta = []
    res = []

    # import pdb
    # pdb.set_trace()
    i = 0
    any_rain = df['Desert_Raining'] + df['Plains_Raining'] + df['Jungle_Raining'] + df['Wetlands_Raining'] > 0
    for r_,r in enumerate(np.array(res_combined).T):

        if tuple(r) in mapping.keys() and (any_rain[r_] == any_rain[r_-1]):
            res.append(mapping[tuple(r)])
            continue
        else:
            mapping[tuple(r)] = i
            theta.append({
                'A': np.array([theta_combined[0][r[0]]['A'][0],
                               theta_combined[1][r[1]]['A'][0],
                               theta_combined[2][r[2]]['A'][0],
                               theta_combined[3][r[3]]['A'][0]]),
                'Cov': np.eye(4)*np.array([theta_combined[0][r[0]]['Cov'][0],
                                           theta_combined[1][r[1]]['Cov'][0],
                                           theta_combined[2][r[2]]['Cov'][0],
                                           theta_combined[3][r[3]]['Cov'][0]])
            })

            i+=1
            res.append(mapping[tuple(r)])

    df['regime'] = res
    df['theta_'+pre.BIOME_NAMES[0]] = [theta[r]['A'][0] for r in res]
    df['theta_'+pre.BIOME_NAMES[1]] = [theta[r]['A'][1] for r in res]
    df['theta_'+pre.BIOME_NAMES[2]] = [theta[r]['A'][2] for r in res]
    df['theta_'+pre.BIOME_NAMES[3]] = [theta[r]['A'][3] for r in res]

    return res, theta, result_assignments, result_params, df, Y


def get_interesting_params(df):

    theta = [df.drop_duplicates(subset=['regime']).theta_Desert.tolist()]
    theta += [df.drop_duplicates(subset=['regime']).theta_Plains.tolist()]
    theta += [df.drop_duplicates(subset=['regime']).theta_Wetlands.tolist()]
    theta += [df.drop_duplicates(subset=['regime']).theta_Jungle.tolist()]
    theta = np.array(theta)

    upper_limit = np.max([np.percentile(theta, 80, axis=1), np.ones(4)*np.percentile(theta, 85)], axis=0)
    lower_limit = np.min([np.percentile(theta, 25, axis=1), np.ones(4)*np.percentile(theta, 20)], axis=0)

    df['interesting_upper_Desert'] = df.apply(lambda row: row.theta_Desert >= upper_limit[0] and (row.Desert_Raining != 1), axis=1).astype(int)
    df['interesting_upper_Plains'] = df.apply(lambda row: row.theta_Plains >= upper_limit[1] and (row.Plains_Raining != 1), axis=1).astype(int)
    df['interesting_upper_Jungle'] = df.apply(lambda row: row.theta_Jungle >= upper_limit[2] and (row.Jungle_Raining != 1), axis=1).astype(int)
    df['interesting_upper_Wetlands'] = df.apply(lambda row: row.theta_Wetlands >= upper_limit[3] and (row.Wetlands_Raining != 1), axis=1).astype(int)

    df['interesting_lower_Desert'] = df.apply(lambda row: row.theta_Desert <= lower_limit[0] and (row.Desert_Raining != 1), axis=1).astype(int)
    df['interesting_lower_Plains'] = df.apply(lambda row: row.theta_Plains <= lower_limit[1] and (row.Plains_Raining != 1), axis=1).astype(int)
    df['interesting_lower_Jungle'] = df.apply(lambda row: row.theta_Jungle <= lower_limit[2] and (row.Jungle_Raining != 1), axis=1).astype(int)
    df['interesting_lower_Wetlands'] = df.apply(lambda row: row.theta_Wetlands <= lower_limit[3] and (row.Wetlands_Raining != 1), axis=1).astype(int)

    df['interesting'] = (df['interesting_lower_Desert']+df['interesting_lower_Plains']+df['interesting_lower_Jungle']+df['interesting_lower_Wetlands']+
                        df['Desert_Raining'] + df['Plains_Raining'] + df['Jungle_Raining'] + df['Wetlands_Raining'])

    return df

def save_for_visualisation(df, path):
    df.to_csv(os.path.join(path, 'input_file.csv'))
