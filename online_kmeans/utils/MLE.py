from scipy.special import (comb, chndtr, entr, rel_entr, xlogy, ive)

# for root finding for discrete distribution ppf, and max likelihood estimation
from scipy import optimize

# for functions of continuous distributions (e.g. moments, entropy, cdf)
from scipy import integrate

# to approximate the pdf of a continuous distribution given its cdf
from scipy.misc import derivative

from numpy import (arange, putmask, ravel, ones, shape, ndarray, zeros, floor,
                   logical_and, log, sqrt, place, argmax, vectorize, asarray,
                   nan, inf, isinf, NINF, empty)

import numpy as np

from scipy.stats import rv_continuous, genextreme, genpareto
from scipy.stats._continuous_distns import genextreme_gen, genpareto_gen

_XMAX = np.finfo(float).max

# This should be rewritten
def argsreduce(cond, *args):
    """Return the sequence of ravel(args[i]) where ravel(condition) is
    True in 1D.

    Examples
    --------
    >>> import numpy as np
    >>> rand = np.random.random_sample
    >>> A = rand((4, 5))
    >>> B = 2
    >>> C = rand((1, 5))
    >>> cond = np.ones(A.shape)
    >>> [A1, B1, C1] = argsreduce(cond, A, B, C)
    >>> B1.shape
    (20,)
    >>> cond[2,:] = 0
    >>> [A2, B2, C2] = argsreduce(cond, A, B, C)
    >>> B2.shape
    (15,)

    """
    newargs = np.atleast_1d(*args)
    if not isinstance(newargs, list):
        newargs = [newargs, ]
    expand_arr = (cond == cond)
    return [np.extract(cond, arr1 * expand_arr) for arr1 in newargs]

def _logpdf(_object, x, *args):
    return log(_object._pdf(x, *args))

def _nnlf_and_penalty(x, args):
    cond0 = ~_object._support_mask(x, *args)
    n_bad = np.count_nonzero(cond0, axis=0)
    if n_bad > 0:
        x = argsreduce(~cond0, x)[0]
    logpdf = _logpdf(_object, x, *args)
    finite_logpdf = np.isfinite(logpdf)
    n_bad += np.sum(~finite_logpdf, axis=0)
    if n_bad > 0:
        penalty = n_bad * log(_XMAX) * 100
        return -np.sum(logpdf[finite_logpdf], axis=0) + penalty
    return -np.sum(logpdf, axis=0)

def _penalized_nnlf(theta, x):
    ''' Return penalized negative loglikelihood function,
    i.e., - sum (log pdf(x, theta), axis=0) + penalty
        where theta are the parameters (including loc and scale)
    '''
    loc, scale, args = _object._unpack_loc_scale(theta)
    if not _object._argcheck(*args) or scale <= 0:
        return inf
    x = asarray((x-loc) / scale)
    n_log_scale = len(x) * log(scale)
    return _nnlf_and_penalty(x, args) + n_log_scale
    
def _reduce_func(self, args, kwds):
    # First of all, convert fshapes params to fnum: eg for stats.beta,
    # shapes='a, b'. To fix `a`, can specify either `f1` or `fa`.
    # Convert the latter into the former.
    if self.shapes:
        shapes = self.shapes.replace(',', ' ').split()
        for j, s in enumerate(shapes):
            val = kwds.pop('f' + s, None) or kwds.pop('fix_' + s, None)
            if val is not None:
                key = 'f%d' % j
                if key in kwds:
                    raise ValueError("Duplicate entry for %s." % key)
                else:
                    kwds[key] = val

    args = list(args)
    Nargs = len(args)
    fixedn = []
    names = ['f%d' % n for n in range(Nargs - 2)] + ['floc', 'fscale']
    x0 = []
    for n, key in enumerate(names):
        if key in kwds:
            fixedn.append(n)
            args[n] = kwds.pop(key)
        else:
            x0.append(args[n])

    if len(fixedn) == 0:
        func = self._penalized_nnlf
        restore = None
    else:
        if len(fixedn) == Nargs:
            raise ValueError(
                "All parameters fixed. There is nothing to optimize.")

        def restore(args, theta):
            # Replace with theta for all numbers not in fixedn
            # This allows the non-fixed values to vary, but
            #  we still call self.nnlf with all parameters.
            i = 0
            for n in range(Nargs):
                if n not in fixedn:
                    args[n] = theta[i]
                    i += 1
            return args

        def func(theta, x):
            newtheta = restore(args[:], theta)
            return self._penalized_nnlf(newtheta, x)

    return x0, func, restore, args

def genextreme_pdf(x, c, loc, scale):
    x = (x-loc)/scale
    if c ==0:
        y = (1 / scale) *np.exp(-np.exp(-x)) * np.exp(-x)
    else:
        y = (1 / scale) * np.exp(-np.power(1 - c * x, 1 / c)) * np.power(1 - c * x, 1 / c - 1)
        y[(c>0)&(x>1/c)] =0.0
    return y

def genextreme_nnlf(theta, x):
    c, loc, scale = theta[0], theta[1], theta[2]
    return -np.sum(np.log(genextreme_pdf(x, c, loc, scale)))

def genextreme_MM(x, c, loc, scale, p):
    y = 0
    for i in range(c.shape[0]):
        y += p[i]*genextreme_pdf(x, c[i], loc[i], scale[i])
    return y

def genextreme_MM_nnlf(theta,x):
    theta = theta.reshape(3,4)
    c, loc, scale, p = theta[:,0], theta[:,1], theta[:,2], theta[:,3]
    return -np.sum(np.log(genextreme_MM(x, c, loc, scale, p)))

def genpareto_pdf(x, c, loc, scale):
    x = (x-loc)/scale
    if c == 0:
        y = (1/scale)*np.exp(-x)
    else:
        y = (1/scale)*np.power(1 + c * x, -1 - 1/c)
        y[(c>0)&(x<0)] = 0.0
        y[(c<0)&((x<0)|(x>-1/c))] = 0.0
    return y


def genpareto_nnlf(theta, x):
    c, loc, scale = theta[0], theta[1], theta[2]
    return -np.sum(np.log(genpareto_pdf(x, c, loc, scale)))


def fit(data):
    start = _object._fitstart(data)
    x0 = list(start)

    optimizer = optimize.fmin

    func = _penalized_nnlf
    vals = optimizer(func, x0, args=(ravel(data),), disp=0)

    vals = tuple(vals)
    return vals

if __name__ == "__main__":
    # rvs = genpareto.rvs(c=0.5, loc=0.5, scale=0.8, size=1000)
    # print(rvs)

    # c, loc, scale = fit(rvs)
    # print(c, loc, scale)

    # c, loc, scale = genpareto.fit(rvs)
    # print(c, loc, scale)

    # p0 = genpareto.pdf(rvs, c=0.5, loc=0.5, scale=0.8)
    # print(p0)

    # p1 = genpareto_pdf(rvs, c=0.5, loc=0.5, scale=0.8)
    # print(p1)
    # print(p0==p1)

    # nnl_sum = genpareto_nnlf(rvs, c=0.5, loc=0.5, scale=0.8)
    # print(nnl_sum)

    # vals = genpareto.fit(rvs)
    # print(vals)

    # optimizer = optimize.fmin
    # vals = optimizer(genpareto_nnlf, [1.0,0,1], args=(ravel(rvs),), disp=0)
    # print(vals)

    # print('#'*20)

    # rvs = genextreme.rvs(c=0.5, loc=0.5, scale=0.8, size=4)
    # print(rvs)

    # p0 = genextreme_pdf(rvs,c=0.5, loc=0.5, scale=0.8)
    # print(p0)

    # p1 = genextreme.pdf(rvs,c=0.5, loc=0.5, scale=0.8)
    # print(p1)


    # vals = genextreme.fit(rvs)
    # print(vals)

    # init_param = genextreme._fitstart(rvs)
    # init_param = list(init_param)
    # optimizer = optimize.fmin
    # vals = optimizer(genextreme_nnlf, init_param, args=(ravel(rvs),), disp=0)
    # print(vals)

    print('#'*20)
    rvs0 = genextreme.rvs(c=0.5, loc=0.5, scale=0.8, size=1000)
    rvs1 = genextreme.rvs(c=0.5, loc=0.5, scale=0.8, size=1000)
    rvs2 = genextreme.rvs(c=0.5, loc=0.5, scale=0.8, size=1000)

    rvs = np.hstack([rvs0,rvs1,rvs2])
    print(rvs)
    print(rvs.shape)

    k=3
    init_param=np.zeros((k,3))
    init_param[0,:] = genextreme._fitstart(rvs0)
    init_param[1,:] = genextreme._fitstart(rvs1)
    init_param[2,:] = genextreme._fitstart(rvs2)
    print(init_param)

    p = np.ones((k,1))/k
    init = np.hstack([init_param, p])
    print(init)

    optimizer = optimize.fmin
    vals = optimizer(genextreme_MM_nnlf, init, args=(ravel(rvs),), disp=0)
    print(vals)
    print(vals.reshape(3,4))


