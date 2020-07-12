import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import cm

import matplotlib
import numpy as np
from scipy.signal import savgol_filter
import scipy as sp
import scipy.stats
from sklearn.neighbors import KernelDensity

#from sklearn.mixture import GaussianMixture

def plot_dist(data):
    
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
    ax.set_title('Dist')

    binwidth=5
    ax.hist(data[:,0], bins = int(len(data)/binwidth))

    plt.show()
    
if __name__ == '__main__':
    time_step = 0
    # Read data
    data = np.genfromtxt('output/vcsmc_particles_{}.csv'.format(time_step),
                         delimiter=',')
    weights = np.genfromtxt('output/vcsmc_weights.csv', delimiter=',')

    observ = [0.292, 0.401, 4.11]
    states = [-0.435, 1.69, 4.00]
    landmark = [1,2,1]
    
    plt.style.use('ggplot')
    COLOR = 'black'
    matplotlib.rcParams['text.color'] = COLOR
    matplotlib.rcParams['axes.labelcolor'] = COLOR
    matplotlib.rcParams['xtick.color'] = COLOR
    matplotlib.rcParams['ytick.color'] = COLOR
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
 
    matplotlib.rcParams.update({'font.size': 22})

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # Hide these grid behind plot objects
    ax.set_axisbelow(True)
    ax.set_xlim((-2,9))
    ax.set_yticklabels([])
    ax.set_xticklabels([-2,0,2,4,6,8])
    bins=len(data[:,1])
    #ax.vlines(0.0, ymin=0, ymax=1., color='gray', linestyle='--')
    #ax.vlines(2.0, ymin=0, ymax=1., color='gray', linestyle='--')

    xhat=np.mean(np.random.choice(data[:,0], 1000, replace=True, p=weights/np.sum(weights)))
    ax.vlines(2, ymin=0, ymax=.2, color='black', linestyle='-')
    #ax.annotate(r'$x$', (-.125,.21))
    
    ax.vlines(states[time_step], ymin=0, ymax=.18, color='black', linestyle='-')    
    ax.vlines(0, ymin=0, ymax=.2, color='black', linestyle='-')    
    ax.vlines(2, ymin=0, ymax=.2, color='black', linestyle='-')    
    ax.vlines(6, ymin=0, ymax=.2, color='black', linestyle='-')    

    ax.annotate(r'$s$', (states[time_step]-.1,.19))
    ax.annotate(r'$\ell_1$', (0-.1,.21))
    ax.annotate(r'$\ell_2$', (2-.1,.21))
    ax.annotate(r'$\ell_3$', (6-.1,.21))
    x_plot = np.linspace(-2,9,1000)

    markersize = 50
    bandwidth=0.7
    # Landmark #1
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data[:,1][:,None])
    log_dens = kde.score_samples(x_plot[:,None])
    ax.fill_between(x_plot, np.exp(log_dens), 0,
                    color='tomato', label=r'$\hat{\ell}_1$', edgecolor='black')
    ax.scatter(data[:,1],-0.05+0*data[:,1],
               marker='+', color='tomato', s=markersize)
    # Landmark #2
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data[:,2][:,None])
    log_dens = kde.score_samples(x_plot[:,None])
    ax.fill_between(x_plot, np.exp(log_dens), 0,
                    color='mediumaquamarine', label=r'$\hat{\ell}_2$', edgecolor='black')
    ax.scatter(data[:,2],-0.05+0*data[:,2],
               marker='+', color='mediumaquamarine', s=markersize)

    # Landmark #3
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data[:,3][:,None])
    log_dens = kde.score_samples(x_plot[:,None])
    ax.fill_between(x_plot, np.exp(log_dens), 0,
                    color='orchid', label=r'$\hat{\ell}_3$', edgecolor='black')
    ax.scatter(data[:,3],-0.05+0*data[:,3],
               marker='+', color='orchid', s=markersize)

    # State
    kde = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(data[:,0][:,None])
    log_dens = kde.score_samples(x_plot[:,None])
    ax.fill_between(x_plot, np.exp(log_dens), 0,
                    color='cornflowerblue', label=r'$\hat{x}$', edgecolor='black')
    ax.scatter(data[:,0],-0.05+0*data[:,0],
               marker='+', color='cornflowerblue', s=markersize)

    ax.set_title(r'$s_{} = {}, \ell_{}, z_{} = {}$'.format(time_step+1,
                                                       states[time_step],
                                                       landmark[time_step],
                                                       time_step+1,
                                                       observ[time_step]))
    #ax.title(r'$s_{2} = {}, z_{\ell_2} = {}$'.format(1.69,0.401))
    #ax.title(r'$s_{3} = {}, z_{\ell_1} = {}$'.format(4,4.11))
    ax.legend(loc='best')
    plt.show()

    
    """
    np.random.seed(1)

    mus =  np.array([[0.], [2.], [6.]])
    sigmas = np.array([[0.1], [0.1], [0.1]]) ** 2
    gmm = GaussianMixture(3)
    gmm.means_ = mus
    gmm.covars_ = sigmas
    gmm.weights_ = np.array([1./3., 1/3., 1./3.])

    #Fit the GMM with random data from the correspondent gaussians
    gaus_samples_1 = np.random.normal(mus[0], sigmas[0], 10).reshape(10,1)
    gaus_samples_2 = np.random.normal(mus[1], sigmas[1], 10).reshape(10,1)
    gaus_samples_3 = np.random.normal(mus[2], sigmas[2], 10).reshape(10,1)
    fit_samples = np.concatenate((gaus_samples_1, gaus_samples_2, gaus_samples_3))
    gmm.fit(fit_samples)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 1, 1000).reshape(1000,1)
    logprob = gmm.score_samples(x)
    pdf = np.exp(logprob)
    #print np.max(pdf) -> 19.8409464401 !?
    ax.plot(x, pdf, '-k')
    """
