import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import cm

import matplotlib 

def plot_table(data, blurs, taus, title='Title', output='wgf_table'):
    #plt.style.use('ggplot')
    COLOR = 'black'
    matplotlib.rcParams['text.color'] = COLOR
    matplotlib.rcParams['axes.labelcolor'] = COLOR
    matplotlib.rcParams['xtick.color'] = COLOR
    matplotlib.rcParams['ytick.color'] = COLOR
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
 
    matplotlib.rcParams.update({'font.size': 18})
    

    n_blur = len(blurs)
    n_taus = len(taus)
    fig, ax = plt.subplots(figsize=(n_blur+2, n_taus+1))
    ax.set_title(title)
    ax.set_xlabel(r"$h$")
    ax.set_ylabel('Final Temperature')

    plt.xlim([0, n_taus])
    plt.ylim([0, n_blur])
    normalized_data = (table_data - np.min(table_data))/(np.max(table_data)-np.min(table_data))
    
    for i in range(n_blur):
        for j in range(n_taus):
            ax.add_patch(
                patches.Rectangle(
                    (j, n_blur - i - 1),  # (x,y)
                    1.0,  # width
                    1.0,  # height
                    facecolor= cm.jet(normalized_data[i,j])
                ))
            ax.text(j+.5, i+.5,
                    '{0:.3g}'.format(table_data[i, j]),
                    ha="center", va="center", color="w")
            
    for i in np.arange(n_taus):
        ax.axvline(i, color='slategray', linestyle=':')
    ax.axvline(n_taus, color='slategray', linestyle=':')
    
    for j in np.arange(n_blur):
        ax.axhline(j, color='slategray', linestyle=':')
    ax.axhline(n_taus, color='slategray', linestyle=':')

#    plt.imshow(table_data)

    # We want to show all ticks...
    ax.set_yticks(np.arange(len(blurs))+0.5)
    ax.set_xticks(np.arange(len(taus))+0.5)

    # ... and label them with the respective list entries
    ax.set_yticklabels(blurs)
    ax.set_xticklabels(taus)


    plt.show()
    plt.savefig(output+'.png')

if __name__ == '__main__':
    # Read data
    data = np.genfromtxt('output/BPFmean_1.csv', skip_header=1, delimiter=',')

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
    xlabels = ['BPF({})'.format(l) for i,l in enumerate(['x',r'$l_1$',r'$l_2$',r'$l_3$'])]
    fig, ax = plt.subplots(figsize=(25, 6))
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # Hide these grid behind plot objects
    ax.set_axisbelow(True)
    ax.set_title('Mean-square Error of Bootstrap Particle Filter')
    ax.set_ylabel(r'$\sqrt{MSE}$')
    locs, labels = plt.xticks()
    plt.setp(ax, xticks=[y + 1 for y in range(len(xlabels))],
             xticklabels=xlabels)

    #    ax.set_xticklabels(methods, fontsize=8)

    violin_parts = ax.violinplot(data, showmeans=False, showmedians=True)
    colors = ['cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue',]

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

