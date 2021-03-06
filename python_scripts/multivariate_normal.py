#!/usr/bin/env python

'''
Need a custom multivariate normal class for fast evaluation of differnet normals for one datapoint in the
forward and backward computations.
'''
import numpy as np
import scipy as sp
from scipy.stats import poisson

def invert(x):
    return np.linalg.inv(np.array(x, dtype=np.float32))

class Poisson:
    def __init__(self, means, covariances):

        self.means = np.array(means)
        mean_shape = self.means.shape

        if len(mean_shape) == 1:
            self.poisson = poisson(self.means)
            def logpdf(self, x_):
                return self.poisson.logpmf(x_)

        elif (mean_shape[0] == mean_shape[1]) and (len(mean_shape) > 2):
            self.poisson = [[poisson(mu) for mu in c] for c in self.means]
            def logpdf(self, x_):
                return [[m.logpmf(x_) for m in c] for c in self.poisson]

        else:
            self.poisson = [poisson(mu) for mu in self.means]
            def logpdf(self, x_):
                return [m.logpmf(x_) for m in self.poisson]

        setattr(self.__class__, 'logpdf', logpdf)

class MultivariateNormal:
    def __init__(self, means, covariances):
        self.means = np.array(means)
        self.cov = np.array(covariances)
        self.const = -0.5*np.log(np.linalg.det(2*np.pi*self.cov))

        mean_shape = self.means.shape

        if len(mean_shape) == 1:
            self.cov_inv = invert(self.cov)
            def logpdf(self, x):
                return self.const - 0.5*np.array((x-self.means).T.dot(self.cov_inv).dot(x-self.means))
        elif (mean_shape[0] == mean_shape[1]) and (len(mean_shape) > 2):
            self.cov_inv = np.array([[invert(self.cov[i][j]) for j in range(mean_shape[0])] for i in range(mean_shape[0])])
            def logpdf(self, x):
                return self.const - 0.5*np.array([[(x-self.means[i][j]).T.dot(self.cov_inv[i][j]).dot(x-self.means[i][j]) for j,_ in enumerate(row)] for i,row in enumerate(self.means)])
        else:
            def logpdf(self, x):
                self.cov_inv = np.array([invert(self.cov[i]) for i in range(mean_shape[0])])
                return self.const - 0.5*np.array([(x-self.means[i]).T.dot(self.cov_inv[i]).dot(x-self.means[i]) for i,_ in enumerate(self.means)])

        setattr(self.__class__, 'logpdf', logpdf)

class Categorical:
    def __init__(self, means, covariances):
        self.means = np.array(means)
        mean_shape = self.means.shape

        if len(mean_shape) == 1:
            self.means -= sp.misc.logsumexp(self.means)
            # normalize the probabilities
            def logpdf(self, x_):
                return np.sum([self.means[int(x)] for x in x_])

        elif (mean_shape[0] == mean_shape[1]) and (len(mean_shape) > 2):
            self.means -= sp.misc.logsumexp(self.means, axis=-1).reshape(self.means.shape[0],self.means.shape[1],-1)
            def logpdf(self, x_):
                return [[np.sum([m[int(x)] for x in x_]) for m in c] for c in self.means]

        else:
            self.means -= sp.misc.logsumexp(self.means, axis=-1).reshape(self.means.shape[0],-1)
            def logpdf(self, x_):
                return [np.sum([m[int(x)] for x in x_]) for m in self.means]

        setattr(self.__class__, 'logpdf', logpdf)
