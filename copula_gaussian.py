import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

class NormalCDF(tfb.Bijector):
  """Bijector that encodes normal CDF and inverse CDF functions.

  We follow the convention that the `inverse` represents the CDF
  and `forward` the inverse CDF (the reason for this convention is
  that inverse CDF methods for sampling are expressed a little more
  tersely this way).

  """
  def __init__(self,loc=0.,scale=1.):
    self.normal_dist = tfd.Normal(loc=loc, scale=scale)
    super(NormalCDF, self).__init__(
        forward_min_event_ndims=0,
        validate_args=False,
        name="NormalCDF")

  def _forward(self, y):
    # Inverse CDF of normal distribution.
    return self.normal_dist.quantile(y)

  def _inverse(self, x):
    # CDF of normal distribution.
    return self.normal_dist.cdf(x)

  def _inverse_log_det_jacobian(self, x):
    # Log PDF of the normal distribution.
    return self.normal_dist.log_prob(x)

class GaussianCopulaTriL(tfd.TransformedDistribution):
  """Takes a location, and lower triangular matrix for the Cholesky factor."""
  def __init__(self, loc, scale_tril):
    super(GaussianCopulaTriL, self).__init__(
        distribution=tfd.MultivariateNormalTriL(
            loc=loc,
            scale_tril=scale_tril),
        bijector=tfb.Invert(NormalCDF()),
        validate_args=False,
        name="GaussianCopulaTriLUniform")

class Concat(tfb.Bijector):
  """This bijector concatenates bijectors who act on scalars.
  
  More specifically, given [F_0, F_1, ... F_n] which are scalar transformations,
  this bijector creates a transformation which operates on the vector 
  [x_0, ... x_n] with the transformation [F_0(x_0), F_1(x_1) ..., F_n(x_n)].
  
  
  NOTE: This class does no error checking, so use with caution.
  
  """
  def __init__(self, bijectors):
    self._bijectors = bijectors
    super(Concat, self).__init__(
        forward_min_event_ndims=1,
        validate_args=False,
        name="ConcatBijector")
    
  @property
  def bijectors(self):
    return self._bijectors
    
  def _forward(self, x):
    split_xs = tf.split(x, len(self.bijectors), -1)
    transformed_xs = [b_i.forward(x_i) for b_i, x_i in zip(
        self.bijectors, split_xs)]
    return tf.concat(transformed_xs, -1)
  
  def _inverse(self, y):
    split_ys = tf.split(y, len(self.bijectors), -1)
    transformed_ys = [b_i.inverse(y_i) for b_i, y_i in zip(
        self.bijectors, split_ys)]
    return tf.concat(transformed_ys, -1)
  
  def _forward_log_det_jacobian(self, x):
    split_xs = tf.split(x, len(self.bijectors), -1)
    fldjs = [
        b_i.forward_log_det_jacobian(x_i, event_ndims=0) for b_i, x_i in zip(
            self.bijectors, split_xs)]
    return tf.squeeze(sum(fldjs), axis=-1)
  
  def _inverse_log_det_jacobian(self, y):
    split_ys = tf.split(y, len(self.bijectors), -1)
    ildjs = [
        b_i.inverse_log_det_jacobian(y_i, event_ndims=0) for b_i, y_i in zip(
            self.bijectors, split_ys)]
    return tf.squeeze(sum(ildjs), axis=-1)

class WarpedGaussianCopula(tfd.TransformedDistribution):
  """Application of a Gaussian Copula on a list of target marginals.
  
  This implements an application of a Gaussian Copula. Given [x_0, ... x_n]
  which are distributed marginally (with CDF) [F_0, ... F_n],
  `GaussianCopula` represents an application of the Copula, such that the
  resulting multivariate distribution has the above specified marginals.
  
  The marginals are specified by `marginal_bijectors`: These are
  bijectors whose `inverse` encodes the CDF and `forward` the inverse CDF.
  """
  def __init__(self, loc, scale_tril, marginal_bijectors):
    super(WarpedGaussianCopula, self).__init__(
        distribution=GaussianCopulaTriL(loc=loc, scale_tril=scale_tril),
        bijector=Concat(marginal_bijectors),
        validate_args=False,
        name="GaussianCopula")