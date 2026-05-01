"""The NGBoost Normal distribution and scores"""

import numpy as np
import scipy as sp
from scipy.special import ndtr
from scipy.stats import norm as dist

from ngboost.distns.distn import RegressionDistn
from ngboost.scores import CRPScore, LogScore

_LOG_2PI = np.log(2.0 * np.pi)


class NormalLogScore(LogScore):
    def score(self, Y):
        z = (Y - self.loc) / self.scale
        return 0.5 * _LOG_2PI + np.log(self.scale) + 0.5 * z**2

    def d_score(self, Y):
        D = np.zeros((len(Y), 2))
        D[:, 0] = (self.loc - Y) / self.var
        D[:, 1] = 1 - ((self.loc - Y) ** 2) / self.var
        return D

    def metric(self):
        FI = np.zeros((self.var.shape[0], 2, 2))
        FI[:, 0, 0] = 1 / self.var
        FI[:, 1, 1] = 2
        return FI

    def _natural_gradient(self, grad, metric):
        nat_grad = np.empty_like(grad)
        nat_grad[:, 0] = grad[:, 0] * self.var
        nat_grad[:, 1] = grad[:, 1] / 2.0
        return nat_grad


class NormalCRPScore(CRPScore):
    def score(self, Y):
        Z = (Y - self.loc) / self.scale
        return self.scale * (
            Z * (2 * sp.stats.norm.cdf(Z) - 1)
            + 2 * sp.stats.norm.pdf(Z)
            - 1 / np.sqrt(np.pi)
        )

    def d_score(self, Y):
        Z = (Y - self.loc) / self.scale
        D = np.zeros((len(Y), 2))
        D[:, 0] = -(2 * sp.stats.norm.cdf(Z) - 1)
        D[:, 1] = self.score(Y) + (Y - self.loc) * D[:, 0]
        return D

    def metric(self):
        I = np.c_[
            2 * np.ones_like(self.var),
            np.zeros_like(self.var),
            np.zeros_like(self.var),
            self.var,
        ]
        I = I.reshape((self.var.shape[0], 2, 2))
        I = 1 / (2 * np.sqrt(np.pi)) * I
        return I


class Normal(RegressionDistn):
    """
    Implements the normal distribution for NGBoost.

    The normal distribution has two parameters, loc and scale, which are
    the mean and standard deviation, respectively.
    This distribution has both LogScore and CRPScore implemented for it.
    """

    n_params = 2
    scores = [NormalLogScore, NormalCRPScore]

    def __init__(self, params):
        super().__init__(params)
        self.loc = params[0]
        self.scale = np.exp(params[1])
        self.var = self.scale**2
        self._dist = None

    def fit(Y):
        m = np.mean(Y)
        s = np.std(Y)
        return np.array([m, np.log(s)])

    def sample(self, m):
        samples = np.random.normal(
            loc=self.loc,
            scale=self.scale,
            size=(m, np.size(self.loc)),
        )
        if np.size(self.loc) == 1:
            return samples.reshape(m)
        return samples

    def mean(self):
        return self.loc

    def logpdf(self, Y):
        z = (Y - self.loc) / self.scale
        return -0.5 * _LOG_2PI - np.log(self.scale) - 0.5 * z**2

    def pdf(self, Y):
        return np.exp(self.logpdf(Y))

    def cdf(self, Y):
        return ndtr((Y - self.loc) / self.scale)

    def ppf(self, q):
        return dist.ppf(q, loc=self.loc, scale=self.scale)

    def rvs(self, size=None, random_state=None):
        if random_state is None:
            return np.random.normal(loc=self.loc, scale=self.scale, size=size)
        if hasattr(random_state, "normal"):
            return random_state.normal(loc=self.loc, scale=self.scale, size=size)
        rng = np.random.default_rng(random_state)
        return rng.normal(loc=self.loc, scale=self.scale, size=size)

    def _scipy_dist(self):
        if self.__dict__.get("_dist") is None:
            self._dist = dist(loc=self.loc, scale=self.scale)
        return self._dist

    def __getattr__(
        self, name
    ):  # gives us Normal.mean() required for RegressionDist.predict()
        frozen_dist = self._scipy_dist()
        if name in dir(frozen_dist):
            return getattr(frozen_dist, name)
        return None

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}


# ### Fixed Variance Normal ###
class NormalFixedVarLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 1))
        D[:, 0] = (self.loc - Y) / self.var
        return D

    def metric(self):
        FI = np.zeros((self.var.shape[0], 1, 1))
        FI[:, 0, 0] = 1 / self.var + 1e-5
        return FI


class NormalFixedVarCRPScore(CRPScore):
    def score(self, Y):
        Z = (Y - self.loc) / self.scale
        return self.scale * (
            Z * (2 * sp.stats.norm.cdf(Z) - 1)
            + 2 * sp.stats.norm.pdf(Z)
            - 1 / np.sqrt(np.pi)
        )

    def d_score(self, Y):
        Z = (Y - self.loc) / self.scale
        D = np.zeros((len(Y), 1))
        D[:, 0] = -(2 * sp.stats.norm.cdf(Z) - 1)
        return D

    def metric(self):
        I = np.c_[2 * np.ones_like(self.var)]
        I = I.reshape((self.var.shape[0], 1, 1))
        I = 1 / (2 * np.sqrt(np.pi)) * I
        return I


class NormalFixedVar(Normal):
    """
    Implements the normal distribution with variance=1 for NGBoost.

    The fixed-variance normal distribution has one parameters, loc which is the mean.
    This distribution has both LogScore and CRPScore implemented for it.
    """

    n_params = 1
    scores = [NormalFixedVarLogScore, NormalFixedVarCRPScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self.loc = params[0]
        self.var = np.ones_like(self.loc)
        self.scale = np.ones_like(self.loc)
        self.shape = self.loc.shape
        self.dist = dist(loc=self.loc, scale=self.scale)

    def fit(Y):
        m, _ = sp.stats.norm.fit(Y)
        return m


# ### Fixed Mean Normal ###
class NormalFixedMeanLogScore(LogScore):
    def score(self, Y):
        return -self.dist.logpdf(Y)

    def d_score(self, Y):
        D = np.zeros((len(Y), 1))
        D[:, 0] = 1 - ((self.loc - Y) ** 2) / self.var
        return D

    def metric(self):
        FI = np.zeros((self.var.shape[0], 1, 1))
        FI[:, 0, 0] = 2
        return FI


class NormalFixedMeanCRPScore(CRPScore):
    def score(self, Y):
        Z = (Y - self.loc) / self.scale
        return self.scale * (
            Z * (2 * sp.stats.norm.cdf(Z) - 1)
            + 2 * sp.stats.norm.pdf(Z)
            - 1 / np.sqrt(np.pi)
        )

    def d_score(self, Y):
        Z = (Y - self.loc) / self.scale
        D = np.zeros((len(Y), 1))
        D[:, 0] = self.score(Y) + (Y - self.loc) * -1 * (2 * sp.stats.norm.cdf(Z) - 1)
        return D

    def metric(self):
        I = np.c_[self.var]
        I = I.reshape((self.var.shape[0], 1, 1))
        I = 1 / (2 * np.sqrt(np.pi)) * I
        return I


class NormalFixedMean(Normal):
    """
    Implements the normal distribution with mean=0 for NGBoost.

    The fixed-mean normal distribution has one parameter, scale which is the standard deviation.
    This distribution has both LogScore and CRPScore implemented for it.
    """

    n_params = 1
    scores = [NormalFixedMeanLogScore, NormalFixedMeanCRPScore]

    # pylint: disable=super-init-not-called
    def __init__(self, params):
        self.loc = np.zeros_like(params[0])
        self.scale = np.exp(params[0])
        self.var = self.scale**2
        self.shape = self.loc.shape
        self.dist = dist(loc=self.loc, scale=self.scale)

    def fit(Y):
        _, s = sp.stats.norm.fit(Y)
        return s
