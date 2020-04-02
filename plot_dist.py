import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import cm

import matplotlib
import numpy as np
from scipy.signal import savgol_filter
import scipy as sp
import scipy.stats

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
    # Read data
    data = np.genfromtxt('output/vcsmc_particles_3.csv', delimiter=',')
    weights = np.genfromtxt('output/vcsmc_weights.csv', delimiter=',')
    

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
    ax.set_xticklabels([-2,r'$\ell_1=0$',r'$\ell_2=2$',4,r'$\ell_3=6$',8])
    bins=len(data[:,1])
    #ax.vlines(0.0, ymin=0, ymax=1., color='gray', linestyle='--')
    #ax.vlines(2.0, ymin=0, ymax=1., color='gray', linestyle='--')

    xhat=np.mean(np.random.choice(data[:,0], 1000, replace=True, p=weights/np.sum(weights)))
    ax.vlines(4, ymin=0, ymax=2., color='black', linestyle='-')
    ax.annotate(r'$x$', (4-.125,2))
    
    ax.vlines(xhat, ymin=0, ymax=2., color='gray', linestyle='--')    
    ax.annotate(r'$\bar{x}$', (xhat-.125,2))
    ax.annotate(r'$\ell_1$', (0,-.1))
    ax.annotate(r'$\ell_2$', (2,-.1))
    ax.annotate(r'$\ell_3$', (6,-.1))
        
    ax.hist(np.random.choice(data[:,1], 1000, replace=True, p=weights/np.sum(weights)),
            bins = np.linspace(np.min(data),np.max(data),30),
            color='tomato',
            edgecolor='black',
            label=r'$\hat{\ell}_1$', density=True)

    ax.hist(np.random.choice(data[:,2], 1000, replace=True, p=weights/np.sum(weights)),
            bins = np.linspace(np.min(data),np.max(data),30),
            color='mediumaquamarine',
            edgecolor='black',
            label=r'$\hat{\ell}_2$', density=True)

    ax.hist(np.random.choice(data[:,3], 1000, replace=True, p=weights/np.sum(weights)),
            bins = np.linspace(np.min(data),np.max(data), 30),
            color='orchid',
            edgecolor='black',
            label=r'$\hat{\ell}_3$', density=True)

    ax.hist(np.random.choice(data[:,0], 1000, replace=True, p=weights/np.sum(weights)),
            bins = np.linspace(np.min(data),np.max(data),30),
            color='cornflowerblue',
            edgecolor='black',
            label=r'$\hat{x}$', density=True)

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
