import tensorflow as tf
import tensorflow_probability as tfp
import copula_gaussian as cg
import matplotlib.pyplot as plt
import seaborn as sb

# x CDF is a Gaussian mixture with two components
x_probs = [0.5, 0.5]
x_locs = [1.0, 2.0]
x_scales = [0.1, 0.1]

# y CDF is a Gaussian
y_loc = 1.0
y_scale = 0.1

# Correlation parameter
rho = 0.5

# Correlation matrix
L_mat = []

# Number of samples
n_samples = 100

# Make CDF bijectors for each marginal
x_cdf = cg.EmpGaussianMixtureCDF(probs=x_probs, locs=x_locs, scales=x_scales)
y_cdf = cg.NormalCDF(loc=y_loc, scale=y_scale)

# Build copula
gc = cg.WarpedGaussianCopula(
    loc=[0., 0., 0.],
    scale_tril=L_mat,
    marginal_bijectors=[x_cdf, y_cdf])

# Do TensorFlow
sess = tf.Session()

samples = gc.sample(n_samples).eval(session=sess)
sb.kdeplot(samples)
plt.show()
