import numpy as np
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

def normalize(data):
    q1 = np.quantile(data, 0.0)
    q3 = np.quantile(data, 1)
    return (data-q1)/(q3-q1)

def plot_violins(data):
        
    plt.style.use('ggplot')
    COLOR = 'black'
    matplotlib.rcParams['text.color'] = COLOR
    matplotlib.rcParams['axes.labelcolor'] = COLOR
    matplotlib.rcParams['xtick.color'] = COLOR
    matplotlib.rcParams['ytick.color'] = COLOR
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
 
    matplotlib.rcParams.update({'font.size': 22})

    # Plot the MSE dist for landmarks
    xlabels = ['BPF({})'.format(l) if i % 2 == 0 else 'VCSMC({})'.format(l) for i,l in enumerate(['x', 'x', r'$\ell_1$',r'$\ell_1$',r'$\ell_2$',r'$\ell_2$',r'$\ell_3$',r'$\ell_3$'])]
    fig, ax = plt.subplots(figsize=(25, 6))
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # Hide these grid behind plot objects
    ax.set_axisbelow(True)
    ax.set_title('Comparision of Filter Posterior Means Estimates')
    ax.set_ylabel(r'$\sqrt{MSE}$')
    #ax.set_ylim((-.05,0.8))
    locs, labels = plt.xticks()
    plt.setp(ax, xticks=[y + 1 for y in range(len(xlabels))],
             xticklabels=xlabels)

    #    ax.set_xticklabels(methods, fontsize=8)
    violin_parts = ax.violinplot(data, showmeans=False, showmedians=True)
    colors = ['cornflowerblue' if i % 2 == 0 else 'tomato' for i in range(len(xlabels))]

    # Make all the violin statistics marks red:
    for partname in ('cbars','cmins','cmaxes','cmedians'):
        vp = violin_parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)

    for i, vp in enumerate(violin_parts['bodies']):
     vp.set_facecolor(colors[i])
     vp.set_edgecolor('slategray')
     vp.set_linewidth(1)
     vp.set_alpha(0.5)

     bplot = ax.boxplot(data,
                        patch_artist=True,
                        showfliers=True,
                        widths=0.05)
     colors = ['cornflowerblue' if i % 2 == 0 else 'tomato' for i in range(len(xlabels))]
    # fill with colors
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for patch, color in zip(bplot['whiskers'], colors):
        plt.setp(patch, color='black')
    for patch, color in zip(bplot['fliers'], colors):
        plt.setp(patch, color='black')
    for patch, color in zip(bplot['means'], colors):
        plt.setp(patch, color='black')
    for patch, color in zip(bplot['medians'], colors):
        plt.setp(patch, color='black')
    for patch, color in zip(bplot['caps'], colors):
        plt.setp(patch, color='black')
    locs, labels = plt.xticks()
    plt.setp(ax, xticks=[y + 1 for y in range(len(xlabels))],
             xticklabels=xlabels)


    plt.show()

def plot_boxes(data):
        
    plt.style.use('ggplot')
    COLOR = 'black'
    matplotlib.rcParams['text.color'] = COLOR
    matplotlib.rcParams['axes.labelcolor'] = COLOR
    matplotlib.rcParams['xtick.color'] = COLOR
    matplotlib.rcParams['ytick.color'] = COLOR
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
 
    matplotlib.rcParams.update({'font.size': 22})

    # Plot the MSE dist for landmarks
    xlabels = ['BPF({})'.format(l) if i % 2 == 0 else 'VCSMC({})'.format(l) for i,l in enumerate(['x', 'x', r'$\ell_1$',r'$\ell_1$',r'$\ell_2$',r'$\ell_2$',r'$\ell_3$',r'$\ell_3$'])]
    fig, ax = plt.subplots(figsize=(25, 6))
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # Hide these grid behind plot objects
    ax.set_axisbelow(True)
    ax.set_title('Comparision of Filter Posterior Means Estimates')
    ax.set_ylabel(r'$\sqrt{MSE}$')
    locs, labels = plt.xticks()
    plt.setp(ax, xticks=[y + 1 for y in range(len(xlabels))],
             xticklabels=xlabels)

    #    ax.set_xticklabels(methods, fontsize=8)
    bplot = ax.boxplot(data,
                       patch_artist=True,
                       showfliers=False)
    colors = ['cornflowerblue' if i % 2 == 0 else 'tomato' for i in range(len(xlabels))]
    # fill with colors
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for patch, color in zip(bplot['whiskers'], colors):
        plt.setp(patch, color='black')
    for patch, color in zip(bplot['fliers'], colors):
        plt.setp(patch, color=color)
    for patch, color in zip(bplot['means'], colors):
        plt.setp(patch, color='black')
    for patch, color in zip(bplot['medians'], colors):
        plt.setp(patch, color='black')
    for patch, color in zip(bplot['caps'], colors):
        plt.setp(patch, color='black')

    plt.show()

def plot_nl_violins(data):
        
    plt.style.use('ggplot')
    COLOR = 'black'
    matplotlib.rcParams['text.color'] = COLOR
    matplotlib.rcParams['axes.labelcolor'] = COLOR
    matplotlib.rcParams['xtick.color'] = COLOR
    matplotlib.rcParams['ytick.color'] = COLOR
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
 
    matplotlib.rcParams.update({'font.size': 22})

    # Plot the MSE dist for landmarks
    xlabels = ['BPF(x)', 'VCSMC(x)']
    fig, ax = plt.subplots(figsize=(25, 6))
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # Hide these grid behind plot objects
    ax.set_axisbelow(True)
    ax.set_title('Comparision of Filter MAP Estimates')
    ax.set_ylabel(r'$\sqrt{MSE}$')
    #ax.set_ylim((-.05,0.8))
    locs, labels = plt.xticks()
    plt.setp(ax, xticks=[y + 1 for y in range(len(xlabels))],
             xticklabels=xlabels)

    #    ax.set_xticklabels(methods, fontsize=8)
    violin_parts = ax.violinplot(data, showmeans=False, showmedians=True)
    colors = ['cornflowerblue' if i % 2 == 0 else 'tomato' for i in range(len(xlabels))]

    # Make all the violin statistics marks red:
    for partname in ('cbars','cmins','cmaxes','cmedians'):
        vp = violin_parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)

    for i, vp in enumerate(violin_parts['bodies']):
     vp.set_facecolor(colors[i])
     vp.set_edgecolor('slategray')
     vp.set_linewidth(1)
     vp.set_alpha(0.5)

     
    plt.show()
    
if __name__ == '__main__':
    # Read data
    bpf_data = np.genfromtxt('output/3door/bpf_rmse_mean_3_99.csv', delimiter=',')
    vsmc_data = np.genfromtxt('output/3door/vcsmc_rmse_mean_3_99.csv', delimiter=',')
    #plot_nl_violins(np.concatenate([bpf_data[:,None],vsmc_data[:,None]], axis=-1))

    #import ipdb; ipdb.set_trace()
    
    data = []
    x_data = []; l_data = []
    x_data.append(bpf_data[:,0][:,None])
    x_data.append(vsmc_data[:,0][:,None])
    for i in range(1,np.shape(vsmc_data)[1]):
        dat = [bpf_data[:,i][:,None], vsmc_data[:,i][:,None]]
        dat = np.concatenate(dat, axis=-1)
        #l_data.append(normalize(dat))
        l_data.append(dat)
    
    x_data = np.concatenate(x_data, axis=-1)
    l_data = np.concatenate(l_data, axis=-1)

    #x_data = normalize(x_data)
    data = np.concatenate([x_data, l_data], axis=-1)

    plot_violins(data)
    #plot_boxes(data)
    #plot_errorbars(data)
