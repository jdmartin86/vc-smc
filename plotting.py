import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sbs

def plot_kde(ref,est,xt_vals,zt_vals):
    plt.figure()
    sbs.kdeplot(ref[:,0], ref[:,1], color='blue')
    sbs.kdeplot(est[:,0], est[:,1], color='green')

    plt.scatter(xt_vals[0,0], xt_vals[0,1], color='red')
    # plt.scatter(xt_vals[1,0], xt_vals[1,1], color='orange')

def plot_dist(ref,est):
    plt.figure()
    sbs.distplot(ref[:,1], color='blue')
    sbs.distplot(est[:,1], color='green')
