import math
import tensorflow as tf
import tensorflow_probability as tfp
import copula_gaussian as cg
import matplotlib.pyplot as plt
import seaborn as sb

"""
gaussian_copula_example.py

Code used to produce the Gaussian Copula examples in Figure 1

For CoRL 2019
"""

# x CDF is a Gaussian mixture with two components
x_probs = [0.5, 0.5]
x_locs = [1.0, 4.0]
x_scales = [0.5, 0.5]

# y CDF is a Gaussian
y_loc = 1.0
y_scale = 1.0

# Correlation parameter
rho1 = -0.9
rho2 = 0.0
rho3 = 0.9

# Correlation matrix
L_mat1 = tf.constant([[1., 0.], [rho1, math.sqrt(1. - rho1**2)]])
L_mat2 = tf.constant([[1., 0.], [rho2, math.sqrt(1. - rho2**2)]])
L_mat3 = tf.constant([[1., 0.], [rho3, math.sqrt(1. - rho3**2)]])

# Number of samples
n_samples = 100000

# Make CDF bijectors for each marginal
x_cdf = cg.EmpGaussianMixtureCDF(ps=x_probs, locs=x_locs, scales=x_scales, n_samples=5000)
y_cdf = cg.NormalCDF(loc=y_loc, scale=y_scale)

# Build copulae
gc1 = cg.WarpedGaussianCopula(
    loc=[0., 0.],
    scale_tril=L_mat1,
    marginal_bijectors=[x_cdf, y_cdf])

gc2 = cg.WarpedGaussianCopula(
    loc=[0., 0.],
    scale_tril=L_mat2,
    marginal_bijectors=[x_cdf, y_cdf])

gc3 = cg.WarpedGaussianCopula(
    loc=[0., 0.],
    scale_tril=L_mat3,
    marginal_bijectors=[y_cdf, x_cdf])

gc4 = cg.WarpedGaussianCopula(
    loc=[0., 0.],
    scale_tril=L_mat3,
    marginal_bijectors=[y_cdf, y_cdf])

# Do TensorFlow
sess = tf.Session()

samples1 = gc1.sample(n_samples).eval(session=sess)
plot1 = sb.jointplot(samples1[:,0], samples1[:,1], kind='kde')
plot1.ax_marg_x.set_xlim([-1,6])
plot1.ax_marg_y.set_ylim([-2,4])

samples2 = gc2.sample(n_samples).eval(session=sess)
plot2 = sb.jointplot(samples2[:,0], samples2[:,1], kind='kde')
plot2.ax_marg_x.set_xlim([-1,6])
plot2.ax_marg_y.set_ylim([-2,4])

samples3 = gc3.sample(n_samples).eval(session=sess)
plot3 = sb.jointplot(samples3[:,0], samples3[:,1], kind='kde')
plot3.ax_marg_x.set_xlim([-2,4])
plot3.ax_marg_y.set_ylim([-1,6])

samples4 = gc4.sample(n_samples).eval(session=sess)
plot4 = sb.jointplot(samples4[:,0], samples4[:,1], kind='kde')
plot4.ax_marg_x.set_xlim([-2,4])
plot4.ax_marg_y.set_ylim([-2,4])
plt.show()
