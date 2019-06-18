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
  def __init__(self):
    self.normal_dist = tfd.Normal(loc=0., scale=1.)
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
