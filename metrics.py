import numpy as np
import tensorflow as tf
import scipy.stats as sps

def kld(p, q, support=1.):
    """
    Compute the approximate KL-divergence between P and Q

    Args:
      p: Tensor of length N where p(i) = p(x_i) for a set of x_i i=1..N
      q: Tensor of length N where q(i) = q(x_i) for a set of x_i i=1..N
      support: Region over which the x_i's lie, used to compute bin sizes

    Returns:
      kld: Scalar equal to sum_i p(i) log (p_i / q_i)
    """
    N = len(p)
    bin_size = support / N
    return bin_size*tf.reduce_sum(p * (tf.log(p) - tf.log(q)))

def angle_difference(a,b):
    """
    Compute the element-wise difference between `a` and `b` on the SO(1) manifold

    Args:
      a: Tensor of angles (not necessarily normalized) (source angle)
      b: Same type as a (target angles)

    Returns:
      diffs: Tensor with same shape as a and b containing element-wise angle
    differences between `a` and `b`, normalized to be in range [-pi,pi]

    """
    diffs = tf.atan2(tf.cos(b - a), tf.sin(b - a))
    return diffs

if __name__ == '__main__':
    """
    Unit tests for metrics
    """

    # Angle difference
    predicted_angles = tf.linspace(-10.0, 10.0, 10)
    true_angles = tf.linspace(10.0, 20.0, 10)
    differences = angle_difference(predicted_angles, true_angles)
    sess = tf.Session()
    differences = sess.run([differences])
    print(differences)

    # KL-divergence
    s1 = 1.0
    s2 = 0.5
    mu1 = 0.0
    mu2 = 1.0

    # Closed form equation for KL-divergence between to Gaussians
    true_kl = tf.log(s2/s1) + (s1**2 + (mu1 - mu2)**2)/(2.*s2**2) - 0.5
    true_kl = sess.run([true_kl])
    print("True KL: ", true_kl)

    # Approximating using the region from [-10, 10]
    xmin = -10
    xmax = 10
    support = xmax - xmin
    xs = np.linspace(-10.0, 10.0, 10000)
    ps = sps.norm.pdf(xs, loc=mu1, scale=s1)
    qs = sps.norm.pdf(xs, loc=mu2, scale=s2)
    approx_kl = kld(ps, qs, support=support)
    approx_kl = sess.run([approx_kl])
    print("Approx KL: ", approx_kl)

