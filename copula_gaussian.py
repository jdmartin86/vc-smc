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

class GaussianMixtureCDF(tfb.Bijector):
  """Bijector that encodes Gaussian mixture CDF and inverse CDF functions.

  We follow the convention that the `inverse` represents the CDF
  and `forward` the inverse CDF (the reason for this convention is
  that inverse CDF methods for sampling are expressed a little more
  tersely this way).

  """
  def __init__(self,ps=[1.], locs=[0.], scales=[1.]):
    self.mixture_dist = tfd.Mixture(
      cat = tfd.Categorical(probs=ps),
      components=[tfd.Normal(loc=loc, scale=scale) for loc,scale in zip(locs, scales)])
    super(GaussianMixtureCDF, self).__init__(
        forward_min_event_ndims=0,
        validate_args=False,
        name="GaussianMixtureCDF")

  def _forward(self, y):
    # Inverse CDF of Gaussian mixture distribution.
    return self.mixture_dist.quantile(y)

  def _inverse(self, x):
    # CDF of Gaussian mixture distribution.
    return self.mixture_dist.cdf(x)

  def _inverse_log_det_jacobian(self, x):
    # Log PDF of the Gaussian mixture distribution.
    return self.mixture_dist.log_prob(x)

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
    # kjd: added "clip by value" to ensure inverse CDF doesn't explode
    # kjd: tested and necessary
    # b_i.forward: [0, 1] -> R
    # TODO: parameterize this clip
    transformed_xs = [tf.clip_by_value(b_i.forward(x_i), clip_value_min=-10.0e5, clip_value_max=10.0e5) for b_i, x_i in zip(
        self.bijectors, split_xs)]
    return tf.concat(transformed_xs, -1)

  # b_i.inverse: R -> [0,1]
  # kjd: tested and necessary
  # TODO: parameterize this epsilon=10.0e-5
  def _inverse(self, y):
    split_ys = tf.split(y, len(self.bijectors), -1)
    transformed_ys = [tf.clip_by_value(b_i.inverse(y_i), clip_value_min=10.0e-3, clip_value_max=1.0-10.0e-3) for b_i, y_i in zip(
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

class EmpiricalCDF(tfb.Bijector):
  """Bijector that encodes normal CDF and inverse CDF functions.

  We follow the convention that the `inverse` represents the CDF
  and `forward` the inverse CDF (the reason for this convention is
  that inverse CDF methods for sampling are expressed a little more
  tersely this way).

  """
  def __init__(self, samples=[0.,.1,.2,.3], interp='nearest'):
    self.dist = tfd.Empirical(samples=samples)
    self.interp = interp
    super(EmpiricalCDF, self).__init__(
        forward_min_event_ndims=0,
        validate_args=False,
        name="EmpiricalCDF")

  def set_params(self, samples):
    # Hack to set params
    self.dist = tfd.Empirical(samples=samples)

  def _forward(self, y):
    # Inverse CDF of empirical distribution.
    y_shape=y.get_shape()
    return tf.reshape(tfp.stats.percentile(self.dist.samples, tf.reshape(y,[-1]), interpolation=self.interp), y_shape)

  def _inverse(self, x):
    # CDF of empirical distribution.
    return self.dist.cdf(x)

  def _inverse_log_det_jacobian(self, x):
    # Log PMF of the empirical distribution.
    return self.dist.log_prob(x)

class EmpGaussianMixtureCDF(tfb.Bijector):
  """Bijector that encodes approx Gaussian mixture CDF and inverse CDF functions.

  We follow the convention that the `inverse` represents the CDF
  and `forward` the inverse CDF (the reason for this convention is
  that inverse CDF methods for sampling are expressed a little more
  tersely this way).

  """
  def __init__(self,ps=[1.], locs=[0.], scales=[1.], n_samples=500, interp='nearest'):
    cat_dist = tfd.Categorical(probs=ps)
    comps = [tfd.Normal(loc=loc, scale=scale) for loc,scale in zip(locs, scales)]
    self.mixture_dist = tfd.Mixture(
      cat = tfd.Categorical(probs=ps),
      components=[tfd.Normal(loc=loc, scale=scale) for loc,scale in zip(locs, scales)])
    self.mu1 = locs[0]
    self.s1 = scales[0]

    self.samples = self.mixture_dist.sample(sample_shape=n_samples)
    self.interp = interp
    super(EmpGaussianMixtureCDF, self).__init__(
        forward_min_event_ndims=0,
        validate_args=False,
        name="EmpGaussianMixtureCDF")

  def _percentile_func(self, vec):
    return tfp.stats.percentile(vec[:-1], 100.0*vec[-1])

  def _forward(self, y):
    """
    Inverse CDF of empirical distribution.
    This is unnecessarily O(len(y)^2)
    Could easily be O(len(y))
    Simple gist of needed implementation (maybe better syntax):
    for j in range(len(y)):
        output[j] = tfp.stats.percentile(self.samples[:,j], 100.*y[j])
    """
    y_shape=y.get_shape()
    # percentile_data = tfp.stats.percentile(self.samples, 100.*tf.reshape(y,[-1]),interpolation='nearest',axis=0)
    if len(self.samples.get_shape().as_list()) > 1:
      percentile_data = tfp.stats.percentile(self.samples, 100.*tf.reshape(y,[-1]),interpolation=self.interp,axis=0)
      return tf.reshape(tf.diag_part(percentile_data), y_shape)
    else:
      percentile_data = tfp.stats.percentile(self.samples, 100.*tf.reshape(y,[-1]),interpolation=self.interp,axis=0)
      return tf.reshape(percentile_data, y_shape)
    # return tfd.Normal(loc=self.mu1, scale=self.s1).quantile(y)

  def _inverse(self, x):
    # CDF of Gaussian mixture distribution.
    if len(self.mixture_dist.batch_shape) < 1:
      return self.mixture_dist.cdf(x)
    else:
      return tf.transpose(self.mixture_dist.cdf(tf.transpose(x)))

  def _inverse_log_det_jacobian(self, x):
    # Log PDF of the Gaussian mixture distribution.
    # return tfd.Normal(tf.transpose(self.mu1), self.s1).log_prob(x)
    if len(self.mixture_dist.batch_shape) < 1:
      return self.mixture_dist.log_prob(x)
    else:
      return tf.transpose(self.mixture_dist.log_prob(tf.transpose(x)))
