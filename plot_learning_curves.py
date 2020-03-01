import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import cm

import matplotlib

import numpy as np
from scipy.signal import savgol_filter
import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    """
    Code obtained from the link below:
    https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), scipy.stats.sem(a, axis=0)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def normalize(dat):
    return (dat - dat.min()) / (dat.max()-dat.min())

def smooth_data(data, window=13, order=3, confidence=0.95):
    mean, lower_bound, upper_bound = mean_confidence_interval(data, confidence=confidence)
    mean = savgol_filter(mean, window, order)
    lower_bound = savgol_filter(lower_bound, window, order)
    upper_bound = savgol_filter(upper_bound, window, order)
    return mean, lower_bound, upper_bound


if __name__ == '__main__':
    # Read data
    dep_data = np.genfromtxt('output/vcsmc_dep_loss_3_4.csv', delimiter=',')
    marg_data = np.genfromtxt('output/vcsmc_dep_loss_3_4.csv', delimiter=',')
    
    alpha = 0.5
    plt.style.use('ggplot')
    COLOR = 'black'
    matplotlib.rcParams['text.color'] = COLOR
    matplotlib.rcParams['axes.labelcolor'] = COLOR
    matplotlib.rcParams['xtick.color'] = COLOR
    matplotlib.rcParams['ytick.color'] = COLOR
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
 
    matplotlib.rcParams.update({'font.size': 22})

    fig, ax = plt.subplots(figsize=(25, 6))
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # Hide these grid behind plot objects
    ax.set_axisbelow(True)
    ax.set_title('Training Curves')
    ax.set_ylabel(r'$ELBO$')

    mean, lower_bound, upper_bound = smooth_data(dep_data)
    steps = np.linspace(1,len(mean),len(mean))
    ax.plot(steps, mean, label='dep', color='cornflowerblue')
    ax.fill_between(steps, lower_bound, upper_bound, alpha=alpha, color='cornflowerblue')

    mean, lower_bound, upper_bound = smooth_data(marg_data)
    steps = np.linspace(1,len(mean),len(mean))
    ax.plot(steps, mean, label='Marginal', color='tomato')
    ax.fill_between(steps, lower_bound, upper_bound, alpha=alpha, color='tomato')

    ax.legend(loc='best')
    plt.show()
