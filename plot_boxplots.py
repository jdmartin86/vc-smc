import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import cm

import matplotlib 


if __name__ == '__main__':
    # Read data
    bpf_data = np.genfromtxt('output/BPFerror_2.csv', skip_header=1, delimiter=',')
    vsmc_data = np.genfromtxt('output/vcsmc_mse_2_99.csv', delimiter=',')

    x_data = []; l_data = []
    x_data.append(bpf_data[:,0][:,None])
    x_data.append(vsmc_data[:,0][:,None])
    for i in range(1,np.shape(vsmc_data)[1]):
        l_data.append(np.sqrt(bpf_data[:,i][:,None]))
        l_data.append(np.sqrt(vsmc_data[:,i][:,None]))
    x_data = np.concatenate(x_data, axis=-1)
    l_data = np.concatenate(l_data, axis=-1)

    plt.style.use('ggplot')
    COLOR = 'black'
    matplotlib.rcParams['text.color'] = COLOR
    matplotlib.rcParams['axes.labelcolor'] = COLOR
    matplotlib.rcParams['xtick.color'] = COLOR
    matplotlib.rcParams['ytick.color'] = COLOR
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
 
    matplotlib.rcParams.update({'font.size': 22})

    # Plot the MSE dist for pose
    xlabels = ['BPF({})'.format(l) if i % 2 == 0 else 'VCSMC({})'.format(l) for i,l in enumerate([r'$\ell_1$',r'$\ell_1$',r'$\ell_2$',r'$\ell_2$',r'$\ell_3$',r'$\ell_3$'])]
    fig, ax = plt.subplots(figsize=(25, 6))
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # Hide these grid behind plot objects
    ax.set_axisbelow(True)
    ax.set_title('Filter Comparision')
    ax.set_ylabel(r'$\sqrt{MSE}$')
    locs, labels = plt.xticks()
    plt.setp(ax, xticks=[y + 1 for y in range(len(xlabels))],
             xticklabels=xlabels)

    #    ax.set_xticklabels(methods, fontsize=8)
    violin_parts = ax.violinplot(l_data, showmeans=False, showmedians=True)
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

