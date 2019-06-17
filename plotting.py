import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sbs

def plot_true(xt_vals, padding=5.0):
    points = np.array([xt_vals[:,0], xt_vals[:,1]]).transpose().reshape(-1,1,2)
    segs = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
    color = np.linspace(0,1,xt_vals.shape[0])
    lc = LineCollection(segs, cmap=plt.get_cmap('viridis'))
    lc.set_array(color)
    plt.gca().add_collection(lc)
    plt.xlim(xt_vals[:,0].min() - padding, xt_vals[:,0].max() + padding)
    plt.ylim(xt_vals[:,1].min() - padding, xt_vals[:,1].max() + padding)
    plt.show()

def plot_kde(ref,est,xt_vals,zt_vals, padding=5.0):
    plt.figure()
    sbs.kdeplot(ref[:,0], ref[:,1], color='blue', bw='scott')
    # sbs.kdeplot(est[:,0], est[:,1], color='green')

    plt.scatter(xt_vals[0,0], xt_vals[0,1], color='red')
    # plt.scatter(xt_vals[1,0], xt_vals[1,1], color='orange')

    plot_true(xt_vals, padding=padding)

def plot_dist(ref,est):
    plt.figure()
    sbs.distplot(ref[:,1], color='blue')
    sbs.distplot(est[:,1], color='green')
    plt.savefig("dist")
    plt.show()
