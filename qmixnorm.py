"""
    Implementation of qmixnorm based on qmixnorm.R from KScorrect R package
"""
import numpy as np
import scipy.stats as sps
import scipy.interpolate as spi
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

def rmixnorm(means, scales, probs, size=1):
    idx = np.random.choice(a=range(len(means)),
                           p=probs,size=size)
    return np.array([np.random.normal(loc=means[i],
                                      scale=scales[i])
                     for i in idx])

def qmixnorm(vals, means, scales, probs, expand=True):
    num_components = len(means)
    assert (np.sum(probs) == 1.0)
    n_samps = 10000
    x = rmixnorm(means,scales,probs,n_samps*num_components)
    x_range = np.ptp(x)
    span = np.linspace(np.min(x) - expand*x_range, np.max(x) + expand*x_range, n_samps)
    # print(span)
    cdf = np.zeros(n_samps)
    for comp in range(num_components):
        cdf += probs[comp] * sps.norm.cdf(span, loc=means[comp], scale=scales[comp])
    # print(np.max(x))
    # print("span shape: ", span.shape)
    # print("CDF shape: ", cdf.shape)
    # print("is nan?: ", np.any(np.isnan(cdf)))
    # print("is inf?: ", np.any(np.isinf(cdf)))
    print(vals)
    quants = tf.py_func(spi.interp1d(cdf, span, kind='slinear'), [vals], np.float64)
    # quants = tfp.math.interp_regular_1d_grid(x=vals,
    #                                          x_ref_min=0.0,
    #                                          x_ref_max=1.0,
    #                                          y_ref=cdf
    #                                          )

    return quants

if __name__ == '__main__':
    means = np.array([0.0, 2.0, 6.0])
    scales = np.array([0.1, 0.1, 0.1])
    probs = np.array([1./3., 1./3., 1./3.])
    samps = rmixnorm(means,scales,probs,size=100)
    vals = np.linspace(1e-10,1-1e-10, 10000)
    # print("Sample shape: ", samps.shape)
    # print(samps)
    sess = tf.Session()
    out = qmixnorm(vals, means, scales, probs).eval(session=sess)
    # print(out)
    plt.plot(vals, out)
    # plt.plot(span, cdf, c="blue")
    plt.show()
