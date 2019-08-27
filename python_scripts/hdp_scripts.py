#!/usr/bin/env python

'''
hdp_scripts.py
Author: Nicholas Hoernle
Date: September 2018
A collection of functions that define a Gibbs sampler for a sticky hierarchical Dirichlet process hidden Markov model (see Emily Fox 2008)
written for the ESSIL use case.
'''

import time
from collections import deque
from operator import itemgetter

import numpy as np
import scipy as sp
import scipy.stats as stats

from multivariate_normal import Categorical
from multivariate_normal import MultivariateNormal
from multivariate_normal import Poisson
import classification_evaluation as hdp_eval
import sys

def invert(x):
    L = np.linalg.inv(np.array(x, dtype=np.float32))
    return L

def get_time(seconds):
    mili_sec = (seconds%1)*1000
    min_ = seconds//60
    sec_ = (seconds - min_*60)
    return "%02d:%02d:%03d" % (min_,sec_,mili_sec)

def progress(stability, time, sample, num_iter, num_chains):
    # TODO: output the progress of the Gibbs sampler here
    sys.stdout.write('{:4}/{}:{:10s} | time: {} | stability: {:4.2f} | #regimes: {}\n'.format(sample, num_iter, '='*(10*sample//num_iter)+'>', get_time(time), stability, num_chains))
    sys.stdout.flush()

def generateTransitionMatrix(dim):
    A = np.random.multivariate_normal(np.zeros(dim), np.eye(dim))
    return A

def generateCovMatrix(dim, scale):
    noise_cov = np.eye(dim)
    return noise_cov

def gaussian_mean(sampler, j, **kwargs):
    return sampler.theta[j]['A']

def gaussian_cov(sampler, j, **kwargs):
    return sampler.theta[j]['Cov']

def gaussian_cov1D(sampler, j, **kwargs):
    return np.sqrt(sampler.theta[j]['Cov'])

def categorical_mean(sampler, j, **kwargs):
    return sampler.theta[j]

def poisson_mean(sampler, j, **kwargs):
    return sampler.theta[j]

def cov_func(theta, t, Y, j):
    return theta[j]['sigma']

def flatten(l):
    return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

def mode_convolution(arr, window=2, dtype=np.int16):

    x = np.zeros_like(arr, dtype=dtype)
    for i,_ in enumerate(x):
        p = np.max([0, i-window])
        e = np.min([len(arr), i+window+1])
        x[i] = stats.mode(arr[p:e])[0][0]
    return x

class sicky_HDP_HMM:

    def __init__(self, Y, starting_params, mean_func, cov_func, likelihood, hmm_type='N-d-gaussian'):
        '''
        Y - the signal that we are dealing with
        starting_params - the params that we are using to initialise the model with
        mean_func - the function to compute the means for the different models
        cov_func - the function to compute the cov for the different models
        hmm_type - ['1-d-gaussian', 'N-d-gaussian', 'categorial']
        '''
        # self.Y = np.array(Y, dtype=np.float32)
        self.Y = Y
        self.L = starting_params['L'] if 'L' in starting_params else 10
        self.log_pi = np.log(starting_params['L']) if 'log_pi' in starting_params else np.log(np.random.dirichlet(alpha=np.ones(self.L), size=self.L))
        self.T = len(Y)
        self.alpha = starting_params['alpha'] if 'alpha' in starting_params else 1
        self.beta = starting_params['beta'] if 'beta' in starting_params else np.random.dirichlet(np.ones(self.L))
        self.kappa = starting_params['kappa'] if 'kappa' in starting_params else 0.1*len(Y)
        self.gamma = starting_params['gamma'] if 'gamma' in starting_params else 1
        self.queue = deque()
        self.threshold = np.log(1e-1)

        if hmm_type == 'N-d-gaussian':
            self.D = np.min(self.Y.shape)
            self.priors = starting_params['priors'] if 'priors' in starting_params else [np.zeros(shape=(self.D)), 3, 1, 1]
            self.theta = [{'A': generateTransitionMatrix(self.D), 'Cov': generateCovMatrix(self.D, 5)} for i in range(self.L)] if 'theta' not in starting_params else starting_params['theta']

            # self.levels = starting_params['levels']
            self.mean_func = gaussian_mean
            self.cov_func = gaussian_cov
            self.likelihood = MultivariateNormal
            self.update_theta = self.__update_theta

            # here we have a mapping of components that are shared amoung regimes
            options = np.arange(self.L).astype(int)

        elif hmm_type == 'categorical':
            self.D = starting_params['D']
            self.priors = starting_params['priors'] if 'priors' in starting_params else [1]

            self.theta = [np.random.dirichlet(alpha=np.ones(self.D)*self.priors[0]) for i in range(self.L)]

            self.mean_func = categorical_mean
            self.cov_func = None
            self.likelihood = Categorical
            self.update_theta = self.update_theta_categorical

        elif hmm_type == 'poisson':
            self.D = 1
            self.priors = starting_params['priors'] if 'priors' in starting_params else [1, 1]

            self.theta = [np.abs(np.random.normal(0,10)) for i in range(self.L)]

            self.mean_func = poisson_mean
            self.cov_func = None
            self.likelihood = Poisson
            self.update_theta = self.update_theta_poisson

        elif hmm_type == '1-d-gaussian':

            self.D = 1
            self.priors = starting_params['priors'] if 'priors' in starting_params else [0, 10, 1, 1]
            self.theta = [{'A': np.array([np.random.normal(0,1)]), 'Cov': generateCovMatrix(self.D, 5)} for i in range(self.L)]

            self.mean_func = gaussian_mean
            self.cov_func = gaussian_cov1D
            self.likelihood = MultivariateNormal
            self.update_theta = self.__update_theta_1d

        else:
            raise BaseException('HMM type must be one of...')
        self.samples = []

    def __backward_step(self, log_pi, obs, likeihood_fn, m_tplus1):
        '''
        The backward message that is passed from zt to zt-1 given by the HMM:
            P(y1:T \mid z_{t-1}, \pi, \theta)
        '''
        messages = (log_pi + likeihood_fn.logpdf(obs) + m_tplus1)
        return sp.misc.logsumexp(messages, axis=1)

    def __backward_algorithm(self, Y, **kwargs):
        '''
        Calculate the backward messages for all time T...1
        '''
        log_pi, theta, L, T = self.log_pi, self.theta, self.L, self.T

        # we have L models and T time steps
        bkwd = np.zeros(shape=(T, L), dtype=np.float32)

        mean_func = self.mean_func
        cov_func = self.cov_func
        likelihood = self.likelihood

        # (a) set the messages T+1,T t
        # we are working in log space
        prev_bkwd = bkwd[-1]

        mu_k_j, sigma_k_j = None, None
        if mean_func != None:
            mu_k_j = np.repeat(np.array([[mean_func(self, j) for j in range(L)]], dtype=np.float32), L, axis=0)
        if cov_func != None:
            sigma_k_j = np.repeat(np.array([[cov_func(self, j) for j in range(L)]], dtype=np.float32), L, axis=0)
        likelihood_fn = likelihood(mu_k_j, sigma_k_j)

        # (b) compute the backward messages
        t = len(Y)-1
        for yt in Y[-1::-1]:

            # evaluate the backward message for each time-step and each assignment for z
            bkwd_val = self.__backward_step(log_pi, yt, likelihood_fn, prev_bkwd)
            prev_bkwd = bkwd_val - sp.misc.logsumexp(bkwd_val)
            bkwd[t] = prev_bkwd
            t -= 1

        return bkwd

    def __forward_step(self, log_pi_ztmin1, obs, likeihood_fn, m_tplus1):
        '''
        The backward message that is passed from zt to zt-1 given by the HMM:
            P(y1:T \mid z_{t-1}, \pi, \theta)
        '''
        obs_likelihood = likeihood_fn.logpdf(obs)

        # if np.max(obs_likelihood) < self.threshold:
        #     self.queue.insert(np.random.choice(np.arange(len(self.queue)+1)), obs)

        prob = log_pi_ztmin1 + obs_likelihood + m_tplus1
        return prob

    def __state_assignments(self, Y, bkwd, **kwargs):
        '''
        Sample the state assignment for z 1 to T and update the sets Y_k accordingly
        '''
        log_pi, theta, L, T = self.log_pi, self.theta, self.L, self.T

        mean_func = self.mean_func
        cov_func = self.cov_func
        likelihood = self.likelihood

        # start with n_{jk} = 0
        n = np.ones(shape=(L,L), dtype=np.int16)
        options = np.arange(0,L,dtype=np.int16)
        z = np.zeros(shape=T, dtype=np.int16)

        sigmas = None
        mus = [mean_func(self, j) for j in range(L)]
        if cov_func != None:
            sigmas = [cov_func(self, j) for j in range(L)]
        likelihood_fn = likelihood(mus, sigmas)

        prob_fk = np.exp(self.__forward_step(np.zeros(L), Y[0], likelihood_fn, bkwd[0]))

        prob_fk /= np.sum(prob_fk)
        zt = np.random.choice(options, p=prob_fk)
        z[0] = zt
        z_tmin1 = zt
        t = 1

        # self.queue.clear()

        for (bkwdt, yt) in zip(bkwd[1:], Y[1:]):

            # (a) compute the probability f_k(yt)
            # pdb.set_trace()
            prob_fk = self.__forward_step(self.log_pi[z_tmin1], yt, likelihood_fn, bkwdt)

            # (b) sample a new z_t
            prob_fk -= sp.misc.logsumexp(prob_fk)
            zt = np.random.choice(options, p=np.exp(prob_fk))

            # (c) increment n
            n[z_tmin1, zt] += 1
            z_tmin1 = zt
            z[t] = zt

            t+=1

        # if len(self.queue) > .25*len(self.Y):
        #     self.threshold += np.log(.95)
        # else:
        #     self.threshold += np.log(1.05)

        # print(len(self.queue), self.threshold)
        return {'z': z, 'n': n}

    #=============================================================
    # some steps for the hdp hmm model
    #=============================================================

    def __step_3(self, state_assignments):

        L, alpha, kappa, beta = self.L, self.alpha, self.kappa, self.beta
        z, J = state_assignments['z'] ,state_assignments['n']

        # initialise m to 0
        m = np.zeros(shape=(L,L), dtype=np.int16)
        J = np.zeros(shape=(L,L), dtype=np.int16)

        # for t in range(1, len(z)):
        #     # counting the number of times we transition from z[t-1] to z[t]
        #     J[z[t-1], z[t]] += 1

        for j in range(0, L):
            for k in range(0, L):

                # initialise n to 0
                # n is the number of customers in the jth restaurant eating the kth dish
                for n in range(1, J[j,k]+1):
                    prob = (alpha*beta[k] + kappa*(j==k))/(n + alpha*beta[k] + kappa*(j==k))
                    x = np.random.binomial(n=1, p=prob)
                    m[j,k] += x

        ##############
        # STEP (b)
        ##############
        rho = kappa/(alpha + kappa)
        w = np.zeros(shape=(L,L), dtype=np.int16)

        for j in range(L):

            if m[j,j] > 0:
                prob = rho/(rho + beta[j]*(1-rho))
                w[j,j] = np.random.binomial(n=m[j,j], p=prob)

        # mbar = mjk if k != j and (m-w) if k==j
        mbar = m - w

        return {'m': m, 'w': w, 'mbar': mbar}


    def __update_beta(self, mbar):
        '''
        Update the global parameter beta according to the specification above
        '''
        gamma, L = self.gamma, self.L

        # TODO: I STILL DON'T KNOW ABOUT THIS ONE
        # mbar_dot = np.sum(mbar, axis=1)
        mbar_dot = np.sum(mbar, axis=0) # I think this one
        p_ = gamma/L + mbar_dot
        # p_[0] = 10
        return np.random.dirichlet(p_)

    def __update_pi(self, state_assignments):

        L, alpha, kappa, beta = self.L, self.alpha, self.kappa, self.beta
        pi = np.zeros(shape=(L,L), dtype=np.float32)
        n = state_assignments['n']

        kappa_ = np.eye(L) * kappa
        for i, pik in enumerate(pi):

            # update the pis
            # pi[i] = np.random.dirichlet(alpha * beta + kappa_[i] + n[:,i]) # this one!!!
            pi[i] = np.random.dirichlet(alpha * beta + kappa_[i] + n[i]) # or this?

        return pi


    def __update_params(self, pi, beta, theta):

        self.log_pi = np.log(pi) # tiny bit for stability
        self.beta = beta
        self.theta = theta

    def __update_theta(self, state_assignments):

        _, Psi, nu, _ = self.priors

        Y, L, theta = self.Y, self.L, self.theta
        z = state_assignments['z']
        # z = mode_convolution(z, window=3)

        # need to update the parameters for each model
        for k in range(L):

            indexes = np.where(z == k)[0]

            if (len(indexes) > 0):

                ykk = Y[indexes]
                nk = len(ykk)

                y_bar = np.mean(ykk, axis=0)
                mean_args = np.argmin(np.abs(self.levels-y_bar), axis=0)
                mu =  self.levels[mean_args, np.arange(0,4)]

                cov_k = (theta[k]['Cov'][np.arange(4), np.arange(4)])


                # posterior sigma values
                nuPost = nk + nu
                PsiPost = Psi + (ykk - mu).T.dot(ykk-mu)

                # sample sigma
                sig = stats.invwishart(nuPost, PsiPost).rvs()

            else:
                # if we don't have any data then we set the posterior to the value of the prior
                sig = stats.invwishart(nu, Psi).rvs()
                mu = self.levels[np.random.choice(list(range(0,self.levels.shape[0])), size=4, replace=True), np.arange(0,4)]
                while tuple(mu) in set([tuple(t['A']) for t in self.theta]):
                    mu = self.levels[np.random.choice(list(range(0,self.levels.shape[0])), size=4, replace=True), np.arange(0,4)]

            theta[k]['Cov'] = sig
            theta[k]['A'] = mu

        return theta

    def __update_theta_CRP(self, state_assignments):
        '''
        Note that the model parameters themselves can be drawn from a DP prior with a Gaussian
        base distribution. We treat each of the sample means as a draw from this prior
        and they could be the same with some probability.
        '''
        mu0, nu, alpha, beta = self.priors

        Y, L, theta = self.Y, self.L, self.theta
        z = state_assignments['z']

        # start naivly by not storing the state of the markov chain
        data = [[],[],[],[]]
        assignments = [[],[],[],[]]
        prior_mu = 0
        prior_sigma = 2
        priors = [0, np.square(1/prior_sigma), 10, .5]
        alpha_post = alpha
        beta_post = beta

        for k in range(L):

            indexes = np.where(z == k)[0]

            ykk = Y[indexes]

            nk = len(ykk)
            y_bar = np.mean(ykk, axis=0)
            samp_var = np.var(ykk, axis=0)

            if nk > 0:

                alpha_post = alpha + nk/2
                beta_post = beta + nk*samp_var/2 + (nk*nu)/(nu + nk) * np.square(y_bar - mu0[dim])/2

            # if we don't have any data then we set the posterior to the value of the prior
            sig = stats.invgamma(a=alpha, scale=beta).rvs()
            T = 1/sig
            theta[k]['Cov'] = np.eye(4)*sig
            theta[k]['A'] = np.random.normal(mu0, np.sqrt(sig/nu))

        # for dim in range(self.D):
        #
        #     dat = np.array(data[dim])
        #     # print(dat)
        #     starting_clusers = assignments[dim]
        #     clusters = copy.deepcopy(starting_clusers)
        #
        #     for i in range(10):
        #         clusters = sample_joint_clusters(priors, dat, clusters, self.L, alpha=10) # don't put too much mass on forcing the clusters
        #
        #     mapping = {}
        #     for c,t in zip(clusters, starting_clusers):
        #         if c not in mapping:
        #             mapping[c] = set()
        #         mapping[c].update([t])
        #
        #     for c in np.unique(clusters):
        #
        #         y = Y[:,dim]
        #
        #         this_assign = copy.deepcopy(z)
        #         for old in mapping[c]:
        #             this_assign[this_assign == old] = -1
        #
        #         indexes = np.where(this_assign == -1)[0]
        #
        #         if len(indexes) > 0:
        #             ykk = y[indexes]
        #
        #             nk = len(ykk)
        #             y_bar = np.mean(ykk)
        #             samp_var = np.var(ykk) if len(ykk) > 1 else 2
        #
        #             # posterior sigma values
        #             alpha_post = alpha + nk/2
        #             beta_post = beta + nk*samp_var/2 + (nk*nu)/(nu + nk) * np.square(y_bar - mu0[dim])/2
        #
        #
        #
        #             mu_post = (nu*mu0[dim] + nk*y_bar)/(nu + nk)
        #             nu_post = nu+nk
        #
        #             mu_samp = np.random.normal(mu_post, np.sqrt(sig/nu_post))
        #
        #             for old in mapping[c]:
        #                 theta[old]['Cov'][dim,dim] = sig
        #                 theta[old]['A'][dim] = mu_samp

        return theta

    #=============================================================
    # put the sampler together
    #=============================================================
    def sample(self, num_iter=100, return_assignments=False, verbose=True, **kwargs):

        Y = self.Y

        assignments = np.zeros(shape=(len(Y), num_iter), dtype=np.int16)
        theta_values = [None for x in range(num_iter)]

        prev_sample = np.zeros(shape=len(Y), dtype=np.int16)
        start = time.time()

        for sample in range(num_iter):
            # print(self.theta)
            bkwd_messages = self.__backward_algorithm(Y)
            state_par = self.__state_assignments(Y, bkwd_messages, **kwargs)

            step3_update = self.__step_3(state_par)

            beta = self.__update_beta(step3_update['mbar'])
            pi = self.__update_pi(state_par)
            theta = self.update_theta(state_par)

            self.__update_params(pi, beta, theta)

            assignments[:, sample] = state_par['z']
            theta_values[sample] = theta

            if sample % 10 == 0:
                seq2_updated, sorted_thetas, hamming_val = hdp_eval.get_hamming_distance(seq1=prev_sample, seq2=state_par['z'])
                end = time.time()

                progress(hamming_val/self.T, (end-start)/10, sample, num_iter, len(np.unique(state_par['z'])))
                start = end

            prev_sample = state_par['z']

        self.assignments = assignments
        self.theta_values = theta_values

        return True

    def get_sample_assignments(self):
        return self.assignments

    def get_sample_theta_params(self):
        return self.theta_values

    def get_posterior_median_assignments(self, fraction=.99):
        res = np.median(self.assignments[:,int(self.assignments.shape[1]*fraction):], axis=1).astype(np.int8)
        return res

    def get_posterior_mean_theta_params(self, fraction=.99):
        theta = []
        num = list(range(-int((1-fraction) * len(self.theta_values)), 0))

        for j in range(self.L):
            theta_A_js = [self.theta_values[ix][j]['A'] for ix in num]
            theta_Cov_js = [self.theta_values[ix][j]['Cov'] for ix in num]
            thet = {'A': np.median(theta_A_js, axis=0), 'Cov': np.median(theta_Cov_js, axis=0)}
            theta.append(thet)

        return theta


#=============================================================
############### Specific parameter updates per model
#=============================================================

    def update_theta_categorical(self, state_assignments):

        Y, L, theta = self.Y, self.L, self.theta
        z = state_assignments['z']

        alpha = self.priors[0]
        vocab_size = self.D

        for j in range(L):
            zs = np.where(z == j)[0]
            observed = np.zeros(vocab_size, dtype=np.int16)
            prior = np.ones(vocab_size, dtype=np.float32)*alpha

            if len(zs > 0):
                Yk = list(itemgetter(*zs)(Y))
                for y in flatten([Yk]):

                    observed[y]+=1

            observed[observed > 10] = 10
            post = observed#/np.max(observed)
            post = post + prior

            theta[j] = np.log(np.random.dirichlet(post))

        return theta

    def update_theta_poisson(self, state_assignments):

        Y, L, theta = self.Y, self.L, self.theta
        z = state_assignments['z']

        alpha, beta = self.priors[0], self.priors[1]

        for j in range(L):

            zs = np.where(z == j)[0]
            X = 0
            n = 0

            if len(zs) > 1:
                Yk = list(itemgetter(*zs)(Y))
                X = np.sum(Yk)
                n = len(Yk)

            theta[j] = np.random.gamma(alpha+X, beta/(beta*n + 1))

        return theta

    def __update_theta_1d(self, state_assignments):

        mu0, nu, alpha, beta = self.priors

        Y, L, theta = self.Y, self.L, self.theta
        z = state_assignments['z']
        # z = mode_convolution(z, window=3)

        # need to update the parameters for each model
        for k in range(self.L):

            indexes = np.where(z == k)[0]
            ykk = Y[indexes]

            nk = len(ykk)

            # if ((nk == 0) and (len(self.queue) > 0)):
            #     ykk = np.array([self.queue.pop()])
            #     nk = len(ykk)
            #     y_bar = ykk[0]
            #     samp_var = 2

            if nk > 0:

                y_bar = np.mean(ykk)
                samp_var = np.var(ykk)

                # posterior sigma values
                alpha_post = alpha + nk/2
                beta_post = beta + nk*samp_var/2 + (nk*nu)/(nu + nk) * np.square(y_bar - mu0)/2

                # sample sigma
                sig = stats.invgamma(a=alpha_post, scale=beta_post).rvs()
                T = 1/sig

                mu_post = (nu*mu0 + nk*y_bar)/(nu + nk)
                nu_post = nu+nk
                mu_samp = np.random.normal(mu_post, np.sqrt(sig/nu_post))

            else:
                # if we don't have any data then we set the posterior to the value of the prior
                sig = stats.invgamma(a=alpha, scale=beta).rvs()
                T = 1/sig
                mu_samp = np.random.normal(mu0, np.sqrt(sig/nu))

            theta[k]['Cov'] = np.array([[sig]])
            theta[k]['A'] = np.array([mu_samp])

        return theta

#=============================================================
# Here are the different sampler models
#=============================================================

def sticky_HMM(Y, starting_params, priors=[0,1,1,10], num_iter=100, verbose=True, return_assignments=False, **kwargs):

    def mean_func(theta, t, Y, j):
        return theta[j]['A']

    def cov_func(theta, t, Y, j):
        return theta[j]['sigma']

    return blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, stats.norm, update_theta, priors, num_iter, return_assignments, verbose, **kwargs)

def sticky_HMM_multi(Y, starting_params, priors=[0,1,1,10], num_iter=100, verbose=True, return_assignments=False, **kwargs):

    def mean_func(theta, t, Y, j):
        return theta[j]['A']

    def cov_func(theta, t, Y, j):
        return theta[j]['sigma']

    return blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, MultivariateNormal, update_theta, priors, num_iter, return_assignments, verbose, **kwargs)

def sticky_HMM_HDP(Y, starting_params, priors, num_iter=100, verbose=True, return_assignments=False, **kwargs):

    def mean_func(theta, t, Y, j):
        return theta[j]

    def cov_func(theta, t, Y, j):
        return 0

    return blocked_Gibbs_for_sticky_HMM_update(Y, starting_params, mean_func, cov_func, Categorical, update_theta_categorical, priors, num_iter, return_assignments, verbose, **kwargs)


def run_multiprocess_inference(y, starting_params, id_, num_iter=100, data_likelihood_type='poisson', **kwargs):

    params = copy.deepcopy(starting_params)
    sampler = sicky_HDP_HMM(y, params, 'hdphmm', 'hdphmm', data_likelihood_type, hmm_type=data_likelihood_type)
    sampler.sample(num_iter=100)

    res, mean = pre.plants_post_process(sampler)
    res = [0 if r < 0.5 else 1 for r in mean]

    return {
        'params': mean,
        'assignments': res,
        'id': id_
    }

def run_multiple_1D_inference(signals, params, numprocs=4):
    from multiprocessing import Pool

    pool = Pool(numprocs)
    kwargs = {'test': 0}

    results = []
    for i,y in enumerate(signals):
        results.append(pool.apply_async(run_multiprocess_inference, args=(y, params, i, ), kwds=kwargs))

    result_params = {}
    result_assignments = {}

    for result in results:
        res_ = result.get()
        result_params[res_['id']] = res_['params']
        result_assignments[res_['id']] = res_['assignments']

    return result_params, result_assignments

def get_posterior_params(priors, sample_stats):
    mu0, nu, alpha, beta = priors
    samp_mean, samp_var, num_data = sample_stats

    mu_post = (nu*mu0 + num_data*samp_mean)/(nu+num_data)
    nu_post = nu+num_data

    alpha_post = alpha+num_data/2
    beta_post = beta + .5*num_data*samp_var + ((num_data*nu)/(nu+num_data))*(((samp_mean-mu0)**2)/2)

    return mu_post, nu_post, alpha_post, beta_post

def get_post_predictive(params):
    mu_post, nu_post, alpha_post, beta_post = params
    return stats.t(df=2*alpha_post, loc=mu_post, scale=beta_post*(nu_post+1)/(nu_post*alpha_post))

def sample_posterior(priors, data, param_priors):

    mu0, sig0, alpha, beta = param_priors

    post_params = priors
    for observation in data:
        post_params = get_posterior_params(post_params, observation)

    mu_post, nu_post, alpha_post, beta_post = post_params

    sigma = stats.invgamma(a=alpha_post, scale=beta_post).rvs()
    tau = np.sqrt(1/sigma)

    tau_post = 1/sig0 + tau
    mu = np.random.normal(mu_post, 1/tau_post)

    return mu, sigma

def sample_joint_clusters(priors, data, clusters, L, alpha=10):

    posteriors = {}
    for i, ((d, S, n), z) in enumerate(zip(data, clusters)):

        # we are considering the new datapoint
        dat = np.concatenate([data[:i], data[i+1:]])
        clus = np.concatenate([clusters[:i], clusters[i+1:]])

        # filter to only the classes that we care about
        indexes = np.where(clus == z)[0]

        if len(indexes) > 0:
            post_params = priors
            for observation in dat[indexes]:
                post_params = get_posterior_params(priors, observation)

            posteriors[z] = get_post_predictive(post_params)

        pred = {}
        # calculate the predictive distributions
        for cluster in np.unique(clus):

            if cluster not in posteriors:
                post_params = priors
                for observation in data[np.where(clusters == cluster)[0]]:
                    post_params = get_posterior_params(priors, observation)

                posteriors[cluster] = get_post_predictive(post_params)

            n = len(dat[np.where(cluster == clus)[0]])
            pred[cluster] = n/(L-1+alpha) * posteriors[cluster].pdf(d)

        # there is a probability that we are looking at a new datapoint
        if len(pred) > 0:
            max_key = 0
            while max_key in pred.keys():
                max_key += 1
        else:
            max_key = 0
        pred[max_key] = alpha/(L-1+alpha) * stats.norm(loc=priors[0], scale=np.sqrt(priors[1])).pdf(d)

        # choose to assign to one of these clusters
        keys = list(pred.keys())

        probs = np.array(list(pred.values()))
        probs /= np.sum(probs)

        assign = np.random.choice(a=keys, p=probs)
        clusters[i] = assign

    return clusters


def plot_inferred_state_seq(state_seq, means, out_folder, data):
    fig, axes = plt.subplots(4, 1, figsize=(25, 10))
    formatter = mpl.ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms)))

    for i, biome in enumerate(BIOMES):
        mean = [means[val][i] for val in state_seq]

        axes[i].plot(data[:,i], alpha=.2)
        axes[i].scatter(np.linspace(0, len(res), num=len(res)), mean, lw=0, s=10)
        axes[i].set_title(biome)
        axes[i].set_ylim([-2,3])

        axes[i].set_xlim([0,8*60])
        axes[i].set_xticks(np.arange(start=0, stop=520, step=60))
        axes[i].set_xticklabels(np.arange(start=0, stop=520, step=60))
        axes[i].xaxis.set_major_formatter(formatter)
        (df[biome]*10).plot(ax=axes[i], alpha=.5)

    fig.tight_layout()
    plt.savefig(os.path.join(out_folder, 'plots/inferred_periods.png'))


def get_interesting_params(mean, threshold_upper=.15, threshold_lower=-.15):
    interesting_information = {}
    for i, biome in enumerate(BIOMES):

        if mean[i] > threshold_upper:
            interesting_information[biome] = ('Water increasing', mean[i])
        if mean[i] < threshold_lower:
            interesting_information[biome] = ('Water decreasing', mean[i])
    return interesting_information


def interesting_conditions(interesting_info, start_time, end_time):
    if len(interesting_info) <= 0:
        return False
    if end_time - start_seconds < 15:  # need the duration to be more than 30s
        return False
    return True


def format_seconds_to_minutes(seconds):
    num_min = seconds // 60
    num_sec = seconds - (num_min * 60)
    return "%02d:%02d" % (num_min, num_sec)



if __name__ == '__main__':
    import argparse
    import pathlib
    import os

    import numpy as np
    import scipy as sp
    import copy
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import scipy.stats as stats

    pd.set_option('display.width', 500)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.notebook_repr_html', True)
    sns.set(rc={'figure.figsize':(15,8)})
    sns.set_style("whitegrid")
    sns.set_context("poster")

    import multivariate_normal, essil_prepocessing as pre, classification_evaluation as hdp_eval

    import sys
    import cv2
    import sys

    # sys.path.insert(0, './scripts')
    import experiment_controller as expctrl
    import inference_controller as ctrl
    import essil_experiment as exp

    c=[[1,0,0,0], [0,1,0,0], [0,0,1,0], [1,1,0,0], [1,0,1,0], [0,1,1,0], [0,0,0,0]]*10

    parser = argparse.ArgumentParser(description='''
    Run an ESSIL raw file through the HDP framework to get model insights and inferences into the
    work that students are completing in Connected Worlds. You can choose noise models from:
    - d dimensional Gaussian where the signal changes are assumed to occur in a correlated manner
    - 1 dimentional Gaussian where the signals are assumed to be uncorrelated.
    ''')

    parser.add_argument('--input_log_file', '-IL', metavar='input_log_file', type=str, nargs=1,
                    help='Input file to the CW .CSV logged data file')

    parser.add_argument('--input_mov_file', '-IM', metavar='input_mov_file', type=str, nargs=1,
                        help='Input file to the CW .MOV logged data file')

    parser.add_argument('--output_dir', '-O', metavar='output_dir', type=str, nargs=1,
                    help='Path to the output directory')

    parser.add_argument('--transform_video', '-T', metavar='trans_vid', type=bool, nargs=1, default=[True],
                        help='Transform the CW .MOV file to a format compatible with the visualisation')


    args = parser.parse_args()

    # create necessary folders and prepare the input
    input_log_file = args.input_log_file[0]
    input_mov_file = args.input_mov_file[0]
    transform_video = args.transform_video[0]
    output_dir = args.output_dir[0]

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    assert os.path.exists(input_log_file)
    assert os.path.exists(input_mov_file)

    orig_stdout = sys.stdout
    # f = open(os.path.join(output_dir, 'output_logs.txt'), 'w')
    # sys.stdout = f
    #
    # try:

    exp.find_or_create_folder(output_dir + '/models/')
    exp.find_or_create_folder(output_dir + '/plots/')

    BIOMES = pre.BIOMES
    df = pre.load_essil_file(input_log_file)

    fig, ax = plt.subplots(1,1,figsize=(20,10))
    ax2 = ax.twinx()

    df[['Floor_Water', 'Waterfall_Water', 'Reservoir_Water']].plot(ax=ax2, ls='--', alpha=.5)
    df[BIOMES].plot(ax=ax, alpha=.9)

    formatter = mpl.ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms)))

    ax.set_xlim([0,8*60])
    ax.set_xticks(np.arange(start=0, stop=520, step=60))
    ax.set_xticklabels(np.arange(start=0, stop=520, step=60))
    ax.xaxis.set_major_formatter(formatter)

    ax.set_title('Water Levels in Different Parts of the Simulation (note the twin axis)')
    ax.set_ylabel('Water amount (biomes)')
    ax2.set_ylabel('Water amount (waterfall, resevoir, floor)')
    ax.set_xlabel('time')
    plt.savefig(os.path.join(output_dir, 'plots/fig1-water.png'))

    water = df[BIOMES].diff()

    if transform_video:
        print('==============================================', file=orig_stdout)
        print('Processing video file ....', file=orig_stdout)
        output_mov = output_dir + '/output_video.mp4'
        cap = cv2.VideoCapture(input_mov_file)
        expctrl.process_global_vid(df, cap, output_mov)
        print('Video process complete', file=orig_stdout)
        print('==============================================', file=orig_stdout)


    d = df.drop_duplicates(subset='_video_seconds')
    data = (d[BIOMES].diff().fillna(0).values / d.seconds.diff().fillna(1).values[:, None]).T

    # sorting out the rain story
    rain_mask = d[[b.replace('Water', 'Raining') for b in BIOMES]].values.astype(int).T == 1

    idx = np.where(~rain_mask, np.arange(rain_mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    data[rain_mask] = data[np.nonzero(rain_mask)[0], idx[rain_mask]]
    data = data.T
    data /= np.percentile(data, 99.0)

    print('', file=orig_stdout)
    print('Getting posterior samples ....', file=orig_stdout)

    # model = exp.generate_full_posterior(data, exp.build_obs_hyperparams2(data), 'geom', output_dir)
    # state_seq, means, orig_state = exp.get_relabeled_state_seq(model, data)
    # exp.plot_state_inference(state_seq, means, data, 'geom', df, [], output_dir)

    state_seq, theta_combined, result_assignments, result_params, df_post, Y = (
        ctrl.run_multithread_inference(algorithm=ctrl.run_inference_HDPHMM, df=df, data=data))

    means = {i:t['A'] for i,t in enumerate(theta_combined)}

    print('Segmented CW data into periods', file=orig_stdout)
    print()
    print('Processing for outfile ... ', file=orig_stdout)

    # TODO: delete these
    print(state_seq, file=orig_stdout)
    print(means, file=orig_stdout)

    res = pre.mode_convolution(state_seq, window=3)
    plot_inferred_state_seq(state_seq, means, output_dir, data)

    print('==============================================', file=orig_stdout)
    print('Finding possible points of interest', file=orig_stdout)
    print('==============================================', file=orig_stdout)

    t = 1
    start_seconds = 0
    state_tmin1 = res[0]
    interesting = np.zeros(len(res))

    while t < len(data):

        state = res[t]

        if (state != state_tmin1) or (t >= len(data)-1):
            end_seconds = t

            interesting_info = get_interesting_params(means[res[t-1]])
            if interesting_conditions(interesting_info, start_seconds, end_seconds):
                interesting[start_seconds:end_seconds] = 1
                # print('Interesting Event')
                # print('Start time:\t{}\t{}'.format(format_seconds_to_minutes(start_seconds), format_seconds_to_minutes(8*60-start_seconds)))
                # print('End time:\t{}\t{}'.format(format_seconds_to_minutes(end_seconds), format_seconds_to_minutes(8*60-end_seconds)))
                # for k,v in interesting_info.items():
                #     print('event:\t\t{event} in {biome} with param {param}'.format(event=v[0], biome=k, param=round(v[1],2)))
                # print()

            start_seconds = t

        state_tmin1 = state
        t += 1

    start, stop = np.zeros(len(res)), np.zeros(len(res))

    last = res[0]
    last_ix = 0

    for i,r in enumerate(res):

        if r == last:
            start[i] = last_ix
            continue

        stop[last_ix:i] = i
        start[i] = i

        last = r
        last_ix = i

    stop[last_ix:] = i
    in_thresh = .2
    out_thresh = -.1

    last_vid_sec = -1
    tops = []
    bottoms = []

    for ((ix, row), (assign)) in zip(df.iterrows(), res):
        means = row[['theta_Desert', 'theta_Plains', 'theta_Jungle', 'theta_Wetlands']].values

        top = [int(m > in_thresh) for m in means]
        bottom = [int(m < out_thresh) for m in means]

        tops.append(top)
        bottoms.append(bottom)

    tops = np.array(tops)
    bottoms = np.array(bottoms)

    df.loc[:, 'interesting_upper_Desert']  = tops[:, 0]
    df.loc[:, 'interesting_upper_Plains']  = tops[:, 1]
    df.loc[:, 'interesting_upper_Jungle']  = tops[:, 2]
    df.loc[:, 'interesting_upper_Wetlands']= tops[:, 3]

    df.loc[:, 'interesting_lower_Desert']  = bottoms[:, 0]
    df.loc[:, 'interesting_lower_Plains']  = bottoms[:, 1]
    df.loc[:, 'interesting_lower_Jungle']  = bottoms[:, 2]
    df.loc[:, 'interesting_lower_Wetlands']= bottoms[:, 3]

    df.loc[:, 'theta_Desert'] = df_post.loc[:, 'theta_Desert']
    df.loc[:, 'theta_Plains'] = df_post.loc[:, 'theta_Plains']
    df.loc[:, 'theta_Jungle'] = df_post.loc[:, 'theta_Jungle']
    df.loc[:, 'theta_Wetlands'] = df_post.loc[:, 'theta_Wetlands']

    df['Desert_specific_regime'] = df_post.loc[:, 'Desert_specific_regime'].astype(int)
    df['Plains_specific_regime'] = df_post.loc[:, 'Plains_specific_regime'].astype(int)
    df['Jungle_specific_regime'] = df_post.loc[:, 'Jungle_specific_regime'].astype(int)
    df['Wetlands_specific_regime'] = df_post.loc[:, 'Wetlands_specific_regime'].astype(int)

    # to_save = df.iloc[df._video_seconds].ffill()
    to_save = df

    to_save[
        ['seconds', '_video_seconds', 'Desert_Water', 'Plains_Water', 'Jungle_Water', 'Wetlands_Water',
         'Desert_Plants', 'Plains_Plants', 'Jungle_Plants', 'Wetlands_Plants', 'Desert_Raining',
         'Plains_Raining', 'Jungle_Raining', 'Wetlands_Raining',
         *[b.replace('Water', lv) for b in pre.BIOMES for lv in ['lv1', 'lv2', 'lv3', 'lv4']],
         'interesting_upper_Desert', 'interesting_upper_Plains', 'interesting_upper_Jungle', 'interesting_upper_Wetlands',
         'interesting_lower_Desert', 'interesting_lower_Plains', 'interesting_lower_Jungle', 'interesting_lower_Wetlands',
         'theta_Desert', 'theta_Plains', 'theta_Jungle', 'theta_Wetlands',
         'Desert_specific_regime', 'Plains_specific_regime', 'Jungle_specific_regime', 'Wetlands_specific_regime'
         ]
    ].to_csv(os.path.join(output_dir, 'model_output.csv'))
    # with open(os.path.join(output_dir, 'model_output.csv'), 'w') as f:
    #     f.write(
    #         'seconds,regime_number,Desert_Water,Plains_Water,Jungle_Water,Wetlands_Water,top1,top2,top3,top4,bottom1,bottom2,bottom3,bottom4,start,stop,vid_sec,interesting,' +
    #         'Desert_Plants,Plains_Plants,Jungle_Plants,Wetlands_Plants,' +
    #         ','.join(*[[b.replace('Water', lv) for b in pre.BIOMES for lv in ['lv1', 'lv2', 'lv3', 'lv4']]]) +
    #         ',description,' + 'Desert_regime,Plains_regime,Jungle_regime,Wetlands_regime,'
    #                          '\n')
    #
    #     last_ix = res[0]
    #
    #     for ((ix, row), (assign), strt, stp, interest) in zip(df.iterrows(), res, start, stop, interesting):
    #         water_flow = np.sort([means[assign][i] for i in range(4)])
    #         water_flow_ix = np.argsort([means[assign][i] for i in range(4)])
    #         top = [i if f > in_thresh else None for i, f in zip(water_flow_ix[::-1], water_flow[::-1])]
    #         bottom = [i if f < out_thresh else None for i, f in zip(water_flow_ix, water_flow)]
    #
    #         f.write(",".join([str(v) for v in [ix, assign, *[row[w] for w in BIOMES], *top, *bottom, strt, stp,
    #                                            row['_video_seconds'], interest,
    #                                            *[row[w.replace('Water', 'Plants')] for w in BIOMES],
    #                                            *[v for v in row[
    #                                                [b.replace('Water', lv) for b in pre.BIOMES for lv in
    #                                                 ['lv1', 'lv2', 'lv3', 'lv4']]].values]
    #             , make_CW_explination_sentence(top, bottom)]]))
    #         f.write(f'{assign},{assign},{assign},{assign}')
    #         f.write("\n")

    print('INFERENCE COMPLETE', file=orig_stdout)
    # except Exception as e:
    #     sys.stdout = orig_stdout
    #     f.close()
    #     print('HAD AN ERROR: ', str(e))
    #     print('INFERENCE FAILED')
    #
    # finally:
    #     sys.stdout = orig_stdout
    #     f.close()
