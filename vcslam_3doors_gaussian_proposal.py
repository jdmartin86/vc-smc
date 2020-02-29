import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow.contrib.distributions as tfd
import plotting
import tensorflow_probability as tfp
# tfb = tfp.bijectors
import correlation_cholesky as cc
from scipy.special import comb
import scipy.stats as sps
import scipy.interpolate as spi
import math
import csv
import time

from bpf import *
from vcsmc import *

import vcslam_agent

# Temporary
import seaborn as sbs
import matplotlib.pyplot as plt

import copula_gaussian as cg

# For evaluation
import metrics

# Remove warnings
tf.logging.set_verbosity(tf.logging.ERROR)

class ThreeDoorsGaussianBPFAgent(vcslam_agent.VCSLAMAgent):
    def __init__(self,
                 target_params,
                 num_steps=3,
                 state_dim=1,
                 num_landmarks=3,
                 landmark_dim=1,
                 latent_dim=None,
                 observ_dim=1,
                 rs=np.random.RandomState(0),
                 prop_scale=0.5,
                 cop_scale=0.5):
        # Target model params
        self.target_params = target_params
        # Number of time steps
        self.num_steps = num_steps
        # Latent state dimensionality (x)
        self.state_dim = state_dim
        # Num landmarks
        self.num_landmarks = num_landmarks
        # Landmark dimensionality (x)
        self.landmark_dim = landmark_dim
        # Latent variable dimensionality
        self.latent_dim = (self.state_dim + self.landmark_dim*self.num_landmarks)
        # A copula over N variables has (N choose 2) correlation parameters
        self.copula_dim = int(comb(self.latent_dim,2))
        # Observation dimensionality (direct observations of x_door)
        self.observ_dim = observ_dim

        self.prop_scale = prop_scale
        self.cop_scale = cop_scale

        # Random state for sampling
        self.rs = rs

    def transition_model(self, x):
        return x + self.target_params[8]

    def measurement_model(self, x):
        return 0.

    def get_dependency_param_shape(self):
        return [self.num_steps, self.copula_dim]

    def get_marginal_param_shape(self):
        return [self.num_steps+1, self.state_dim*4]

    def init_marg_params(self):
        """
        Marginal params for this problem are the means of the Gaussians at each
        time step (so 3 gaussians * T time steps) as well as the landmark
        location means (so num landmarks = 3) which we place at the T+1 time step

        """
        # Test initialization with prior landmark means
        marg_params = np.vstack([np.array([np.array([self.prop_scale*self.rs.randn(3*self.state_dim)]).ravel() for t in range(self.num_steps)], dtype=np.float32),
                                 np.array([0.0, 2.0, 6.0], dtype=np.float32)])
        return marg_params

    def init_dependency_params(self):
        # Correlation params
        
        copula_params = np.array([np.array(self.cop_scale * self.rs.randn(self.copula_dim)).ravel() for t in range(self.num_steps)], dtype=np.float32)
        return copula_params

    def generate_data(self):
        init_mean, init_var, lm1_prior_mean, lm1_prior_var, lm2_prior_mean, lm2_prior_var,lm3_prior_mean, lm3_prior_var, motion_mean, motion_var, meas_var = self.target_params
        Dx = init_mean.get_shape().as_list()[0]
        #Dz = R.get_shape().as_list()[0]

        x_true = []
        #z_true = []

        for t in range(self.num_steps):
            if t > 0:
                x_true.append(tf.transpose(tfd.MultivariateNormalFullCovariance(loc=tf.transpose(self.transition_model(x_true[t-1])),
                                                                   covariance_matrix=Q).sample(seed=self.rs.randint(0,1234))))
            else:
                x_sample = init_mean
                x_true.append(x_sample)

        return x_true#, z_true

    def sim_proposal(self, t, x_prev, observ, proposal_params):
        init_mean, init_var, lm1_prior_mean, \
        lm1_prior_var, lm2_prior_mean, lm2_prior_var, \
        lm3_prior_mean, lm3_prior_var, motion_mean, \
        motion_var, meas_var = self.target_params
        
        num_particles = x_prev.get_shape().as_list()[0]
        prop_copula_params = proposal_params[0]
        prop_marg_params = proposal_params[1]

        mu1t = prop_marg_params[t,0]
        
        l1m = prop_marg_params[self.num_steps, 0]
        l2m = prop_marg_params[self.num_steps, 1]
        l3m = prop_marg_params[self.num_steps, 2]

        if t == 0:
            mu1 = mu1t
            x_scale = np.float32(np.squeeze(np.sqrt(init_var)))
        else:
            transition = tf.transpose(self.transition_model(tf.transpose(x_prev[:,0])))
            mu1 = mu1t + transition
            x_scale = np.float32(np.squeeze(np.sqrt(motion_var)))

        x_cdf = cg.NormalCDF(loc=mu1, scale=x_scale)
        l1_cdf = cg.NormalCDF(loc=l1m, scale=np.sqrt(np.float32(lm1_prior_var)))
        l2_cdf = cg.NormalCDF(loc=l2m, scale=np.sqrt(np.float32(lm2_prior_var)))
        l3_cdf = cg.NormalCDF(loc=l3m, scale=np.sqrt(np.float32(lm3_prior_var)))

        # Build a copula (can also store globally if we want) we would just
        #  have to modify self.copula.scale_tril and
        #  self.copula.marginal_bijectors in each iteration NOTE: I add
        #  tf.eye(3) to L_mat because I think the diagonal has to be > 0
        gc = cg.WarpedGaussianCopula(
            loc=[0., 0., 0., 0.],
            scale_tril=tf.eye(4),
            marginal_bijectors=[x_cdf, l1_cdf, l2_cdf, l3_cdf])

        return gc.sample(x_prev.get_shape().as_list()[0])

    def log_normal(self, x, mu, Sigma):
        dim = np.shape(Sigma)[0]
        sign, logdet = tf.linalg.slogdet(Sigma)
        log_norm = -0.5*dim*np.log(2.*np.pi) - 0.5*logdet
        Prec = tf.dtypes.cast(tf.linalg.inv(Sigma), dtype=tf.float32)
        first_term = x - mu
        second_term = tf.transpose(tf.matmul(Prec, tf.transpose(x-mu)))
        ls_term = -0.5*tf.reduce_sum(first_term*second_term,1)
        return tf.cast(log_norm, dtype=tf.float32) + tf.cast(ls_term, dtype=tf.float32)

    def log_target(self, t, x_curr, x_prev, observ):
        # init_pose, init_cov, A, Q, C, R = self.target_params
        init_mean, init_var, lm1_prior_mean, \
        lm1_prior_var, lm2_prior_mean, lm2_prior_var,\
        lm3_prior_mean, lm3_prior_var, motion_mean, \
        motion_var, meas_var = self.target_params
        
        x_prev_samples = tf.transpose(tf.gather_nd(tf.transpose(x_prev),[[0]]))
        x_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[0]]))
        l1_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[1]]))
        l2_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[2]]))
        l3_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[3]]))
        logF = 0.0
        logG = 0.0
        if t > 0:
            logF = self.log_normal(x_samples,
                                   tf.transpose(self.transition_model(tf.transpose(x_prev_samples))), motion_var)
        if t == 0 or t == 1:
            log_mixture_components = [tf.log(1./3.) + self.log_normal(x_samples, l1_samples, meas_var),
                                      tf.log(1./3.) + self.log_normal(x_samples, l2_samples, meas_var),
                                      tf.log(1./3.) + self.log_normal(x_samples, l3_samples, meas_var)]
            logG = tf.reduce_logsumexp(log_mixture_components, axis=0)
        logH = self.log_normal(l1_samples, lm1_prior_mean, lm1_prior_var) + \
               self.log_normal(l2_samples, lm2_prior_mean, lm2_prior_var) + \
               self.log_normal(l3_samples, lm3_prior_mean, lm3_prior_var)
        return logF + logG + logH

    def log_weights(self, t, x_curr, x_prev, observ, proposal_params):
        target_log = self.log_target(t, x_curr, x_prev, observ)
        target_log = tf.debugging.check_numerics(target_log, "Target log error")
        prop_log = self.log_proposal(t, x_curr, x_prev, observ, proposal_params)
        prop_log = tf.debugging.check_numerics(prop_log, "Proposal log error")
        return target_log - prop_log

if __name__ == '__main__':
    # List available devices
    #local_device_protos = device_lib.list_local_devices()
    #print([x.name for x in local_device_protos])
    # Optionally use accelerated computation
    # with tf.device("/device:XLA_CPU:0"):

    # Number of steps for the trajectory
    num_steps = 3
    # Number of particles to use during training
    num_train_particles = 100
    # Number of particles to use during SMC query
    num_query_particles = 2000
    # Number of iterations to fit the proposal parameters
    num_train_steps = 4000
    # Learning rate for the marginal
    lr_m = 0.1
    # Learning rate for the copula
    lr_d = 0.001
    # Number of random seeds for experimental trials
    num_seeds = 10
    # Number of samples to use for plotting
    num_samps = 2000
    # Proposal initial scale
    prop_scale = 2.0
    # Copula initial scale
    cop_scale = 0.1

    # True target parameters
    lm1_prior_mean = np.array([[0.]], dtype=np.float32)
    lm1_prior_var = np.array([[0.01]], dtype=np.float32)
    lm2_prior_mean = np.array([[2.0]], dtype=np.float32)
    lm2_prior_var = np.array([[0.01]], dtype=np.float32)
    lm3_prior_mean = np.array([[6.0]], dtype=np.float32)
    lm3_prior_var = np.array([[0.01]], dtype=np.float32)
    motion_mean = np.array([[2.0]], dtype=np.float32)
    motion_var = np.array([[0.1]], dtype=np.float32)
    meas_var = np.array([[0.1]], dtype=np.float32)
    init_mean = np.array([[0.]], dtype=np.float32)
    init_var = np.array([[5.]], dtype=np.float32)
    target_params = [init_mean, init_var,
                     lm1_prior_mean, lm1_prior_var,
                     lm2_prior_mean, lm2_prior_var,
                     lm3_prior_mean, lm3_prior_var,
                     motion_mean, motion_var,
                     meas_var]


    # Create the agent
    rs = np.random.RandomState(1)# This remains fixed for the ground truth
    td_agent = ThreeDoorsGaussianBPFAgent(target_params=target_params,
                                          rs=rs,
                                          num_steps=num_steps,
                                          prop_scale=prop_scale,
                                          cop_scale=cop_scale)
    
    truth = np.array([[0, 0, 2, 4],
                      [2, 0, 2, 6],
                      [4, 0, 2, 6]], np.int64)
    np.savetxt('output/trajectory_ref.txt', truth, delimiter=',')
    
    zt_vals = None

    trial_mean_errors = []; trial_map_errors = []; trial_means = []
    for seed in range(num_seeds):
        start = time.time()
        tf.reset_default_graph()
        tf.set_random_seed(seed)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./logs', sess.graph)
            vcs = VCSLAM(sess=sess,
                         vcs_agent = td_agent,
                         observ = zt_vals,
                         num_particles = num_train_particles,
                         num_train_steps = num_train_steps,
                         lr_d = lr_d,
                         lr_m = lr_m,
                         summary_writer = writer)

            # Initialize the model
            dep_params = td_agent.init_dependency_params()
            marg_params = td_agent.init_marg_params()
            proposal_params = [dep_params, marg_params]

            # Sample the model
            particles, map_traj = vcs.sim_q(proposal_params,
                                            target_params,
                                            zt_vals,
                                            td_agent,
                                            num_samples=num_train_particles)
            particles, map_traj = sess.run([particles, map_traj])
        particles = np.squeeze(np.array(particles))
        #print('Estimated trajectory samples: {}'.format(particles[-1,:]))

        # record trial data
        mean_traj = np.mean(particles[-1,:,:], 0)
        trial_mean_errors.append(np.sqrt((truth[num_steps-1,:] - mean_traj)**2))
        trial_map_errors.append(np.sqrt((truth[num_steps-1,:] - map_traj[-1,:])**2))
        trial_means.append(mean_traj)
            
        # Print elapsed time for the trial
        end = time.time()
        graph_vars = [n.name for n in tf.get_default_graph().as_graph_def().node]
        print("Trial #{}: Elapsed time {}, Graph nodes {}".format(seed,
                                                                  end - start,
                                                                  len(graph_vars)))
        #save the data
        np.savetxt('output/bpf_rmse_mean_{}_{}.csv'.format(num_steps,seed), trial_mean_errors, delimiter=',')
        np.savetxt('output/bpf_rmse_map_{}_{}.csv'.format(num_steps,seed), trial_map_errors, delimiter=',')
        np.savetxt('output/bpf_mean_{}_{}.csv'.format(num_steps,seed), trial_means, delimiter=',')




