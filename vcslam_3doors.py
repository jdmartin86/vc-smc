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
import math
import csv

from vcsmc import *
import vcslam_agent

# Temporary
import seaborn as sbs
import matplotlib.pyplot as plt

import copula_gaussian as cg
import metrics

import time


# Remove warnings
tf.logging.set_verbosity(tf.logging.ERROR)

class ThreeDoorsAgent(vcslam_agent.VCSLAMAgent):
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
        d1 = np.sqrt((self.target_params[2] - x)**2)
        d2 = np.sqrt((self.target_params[2] - x)**2)
        d3 = np.sqrt((self.target_params[2] - x)**2)
        
        return np.min([d1,d2,d3])

    def get_dependency_param_shape(self):
        return [self.num_steps, self.copula_dim]

    def get_marginal_param_shape(self):
        return [self.num_steps+1, self.state_dim*3]

    def init_marg_params(self):
        """
        Marginal params for this problem are the means of the Gaussians at each
        time step (so 3 gaussians * T time steps) as well as the landmark
        location means (so num landmarks = 3) which we place at the T+1 time step

        """
        # Test initialization with prior landmark means
        marg_params = np.vstack([np.array([np.array([np.random.normal(loc=0.1, scale=0.001, size=3)]).ravel() for t in range(self.num_steps)]),
                                 np.array([np.random.normal(0.,2.),np.random.normal(2.,2.),np.random.normal(6.,2.)], dtype=np.float32)])
        return marg_params

    def init_dependency_params(self):
        # Correlation params
        copula_params = np.array([np.array(self.cop_scale * self.rs.randn(self.copula_dim)).ravel() for t in range(self.num_steps)])
        return copula_params

    def sim_proposal(self, t, x_prev, observ, proposal_params):
        with tf.variable_scope("sim_proposal", reuse=tf.AUTO_REUSE):
            
            init_mean, init_var, lm1_prior_mean,\
            lm1_prior_var, lm2_prior_mean, lm2_prior_var,\
            lm3_prior_mean, lm3_prior_var, motion_mean, \
            motion_var, meas_var = self.target_params
            
            num_particles = x_prev.get_shape().as_list()[0]
            prop_copula_params = proposal_params[0]
            prop_marg_params = proposal_params[1]
            
            mu1t = prop_marg_params[t, 0]
            mu2t = prop_marg_params[t, 1]
            mu3t = prop_marg_params[t, 2]

            l1m = prop_marg_params[self.num_steps, 0]
            l2m = prop_marg_params[self.num_steps, 1]
            l3m = prop_marg_params[self.num_steps, 2]
            
            ps = np.array([1./3., 1./3., 1./3.], dtype=np.float32)

            if t == 0:
                mu1 = mu1t
                mu2 = mu2t
                mu3 = mu3t
                x_scale = np.float32(np.squeeze(np.sqrt(init_var)))
            else:
                transition = tf.transpose(self.transition_model(tf.transpose(x_prev[:,0])))
                mu1 = mu1t + transition
                mu2 = mu2t + transition
                mu3 = mu3t + transition

                ps = np.repeat(ps[None,:], num_particles, axis=0)

                mu1 = mu1[:,0]
                mu2 = mu2[:,0]
                mu3 = mu3[:,0]

                x_scale = np.float32(np.squeeze(np.sqrt(motion_var)))

            r_vec = prop_copula_params[t,:]
            L_mat = cc.CorrelationCholesky().forward(r_vec)

            # Defining the marginal CDFs 
            x_cdf = cg.EmpGaussianMixtureCDF(ps=ps,
                                             locs=[mu1, mu2, mu3],
                                             scales=[x_scale, x_scale, x_scale])

            l1_cdf = cg.NormalCDF(loc=l1m, scale=np.sqrt(np.float32(lm1_prior_var)))
            l2_cdf = cg.NormalCDF(loc=l2m, scale=np.sqrt(np.float32(lm2_prior_var)))
            l3_cdf = cg.NormalCDF(loc=l3m, scale=np.sqrt(np.float32(lm3_prior_var)))

            # Build a copula (can also store globally if we want)
            #NOTE: I add tf.eye(3) to L_mat because I think the diagonal has to be > 0
            gc = cg.WarpedGaussianCopula(
                loc=[0., 0., 0., 0.],
                scale_tril=L_mat,
                marginal_bijectors=[x_cdf, l1_cdf, l2_cdf, l3_cdf])
            
            return gc.sample(x_prev.get_shape().as_list()[0])

    def log_proposal(self,t,x_curr,x_prev,observ,proposal_params):
        """
        Log probability from the state-component copula
        """
        prop_copula_params, prop_marg_params = proposal_params

        init_mean, init_var, lm1_prior_mean, \
        lm1_prior_var, lm2_prior_mean, lm2_prior_var, \
        lm3_prior_mean, lm3_prior_var, motion_mean, \
        motion_var, meas_var = self.target_params

        T = self.num_steps
        num_particles = x_prev.get_shape().as_list()[0]
        prop_copula_params = proposal_params[0]
        prop_marg_params = proposal_params[1]

        mu1t = prop_marg_params[t,0]
        mu2t = prop_marg_params[t,1]
        mu3t = prop_marg_params[t,2]

        l1m = prop_marg_params[T,0]
        l2m = prop_marg_params[T,1]
        l3m = prop_marg_params[T,2]

        if t == 0:
            mu1 = mu1t
            mu2 = mu2t
            mu3 = mu3t
        if t > 0:
            mu1 = mu1t + tf.transpose(self.transition_model(tf.transpose(x_prev[:,0])))
            mu2 = mu2t + tf.transpose(self.transition_model(tf.transpose(x_prev[:,0])))
            mu3 = mu3t + tf.transpose(self.transition_model(tf.transpose(x_prev[:,0])))

        # Copula params are defined over the reals, but we want a correlation matrix
        # So we use a sigmoid map to take reals to the range [-1,1]
        # r_vec = (1. - tf.exp(-prop_copula_params[t,:]))/(1. + tf.exp(-prop_copula_params[t,:])) # should be length
        # use correlationcholesky to do the right thing
        r_vec = prop_copula_params[t,:]
        L_mat = cc.CorrelationCholesky().forward(r_vec)

        # Marginal bijectors will be the CDFs of the univariate marginals Here
        # these are normal CDFs and GaussianMixtureCDF
        # x_cdf = cg.GaussianMixtureCDF(ps=[1.], locs=[mu1, mu2, mu3], scales=[tf.sqrt(motion_var), tf.sqrt(motion_var), tf.sqrt(motion_var)])

        # ps = tf.transpose(tf.constant([[1./3.], [1./3.], [1./3.]]))
        ps = [1./3., 1./3., 1./3.]
        # ps = [0.25, 0.25, 0.25, 0.25]
        # ps = [1.0]
        # ps = tf.Print(ps, [ps], "P values")

        if t == 0:
            # x_scale = tf.sqrt(4.0*meas_var)
            x_scale = tf.sqrt(init_var)
        if t > 0:
            x_scale = tf.sqrt(motion_var)

        if t == 0:
            ps = [1./3., 1./3., 1./3.]
        if t > 0:
            ps = tf.constant([1./3., 1./3., 1./3.])*tf.ones_like(mu1)
            mu1 = mu1[:,0]
            mu2 = mu2[:,0]
            mu3 = mu3[:,0]

        x_cdf = cg.EmpGaussianMixtureCDF(ps=ps,
                                         locs=[mu1, mu2, mu3],
                                         scales=[x_scale[0,0], x_scale[0,0], x_scale[0,0]])

        l1_cdf = cg.NormalCDF(loc=l1m, scale=tf.sqrt(lm1_prior_var))
        l2_cdf = cg.NormalCDF(loc=l2m, scale=tf.sqrt(lm2_prior_var))
        l3_cdf = cg.NormalCDF(loc=l3m, scale=tf.sqrt(lm3_prior_var))

        # Build a copula (can also store globally if we want) we would just
        #  have to modify self.copula.scale_tril and
        #  self.copula.marginal_bijectors in each iteration NOTE: I add
        #  tf.eye(3) to L_mat because I think the diagonal has to be > 0
        gc = cg.WarpedGaussianCopula(
            loc=[0., 0., 0., 0.],
            scale_tril=L_mat, # TODO This is currently just eye(3), use L_mat!
            marginal_bijectors=[
                x_cdf,
                l1_cdf,
                l2_cdf,
                l3_cdf])

        return gc.log_prob(x_curr)

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
        init_mean, init_var, lm1_prior_mean, lm1_prior_var, lm2_prior_mean, lm2_prior_var, lm3_prior_mean, lm3_prior_var, motion_mean, motion_var, meas_var = self.target_params
        x_prev_samples = tf.transpose(tf.gather_nd(tf.transpose(x_prev),[[0]]))
        x_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[0]]))
        l1_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[1]]))
        l2_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[2]]))
        l3_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[3]]))

        logF = 0.0
        logG = 0.0
        if t > 0:
            logF = self.log_normal(x_samples, tf.transpose(self.transition_model(tf.transpose(x_prev_samples))), motion_var)
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
    # local_device_protos = device_lib.list_local_devices()
    # print([x.name for x in local_device_protos])
    # Optionally use accelerated computation
    # with tf.device("/device:XLA_CPU:0"):

    # Number of steps for the trajectory
    num_steps = 3 # Must be greater than 1
    # Number of particles to use during training
    num_train_particles = 100
    # Number of iterations to fit the proposal parameters
    num_train_steps = 1000
    # Learning rate for the marginal
    lr_m = 0.1
    # Learning rate for the copula
    lr_d = 0.1
    # Number of random seeds for experimental trials
    num_seeds = 100
    # Number of samples to use for plotting
    num_samps = 2000
    # Proposal initial scale
    prop_scale = 2.0
    # Copula initial scale
    cop_scale = 0.1

    # True target parameters
    lm1_prior_mean = np.array([[0.]], dtype=np.float32)
    lm2_prior_mean = np.array([[2.]], dtype=np.float32)
    lm3_prior_mean = np.array([[6.]], dtype=np.float32)
    lm1_prior_var = np.array([[0.01]], dtype=np.float32)
    lm2_prior_var = np.array([[0.01]], dtype=np.float32)
    lm3_prior_var = np.array([[0.01]], dtype=np.float32)
    motion_mean = np.array([[2.]], dtype=np.float32)
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
    td_agent = ThreeDoorsAgent(target_params=target_params,
                               rs=rs,
                               num_steps=num_steps,
                               prop_scale=prop_scale,
                               cop_scale=cop_scale)

    # Generate observations TODO: change to numpy implementation
    # x_true, z_true = td_agent.generate_data()
    # xt_vals, zt_vals = sess.run([x_true, z_true])
    zt_vals = None

    truth = np.array([[0, 0, 2, 4],
                      [2, 0, 2, 6],
                      [4, 0, 2, 6]], np.int64)
    np.savetxt('output/trajectory_ref.txt', truth, delimiter=',')

    trial_mean_errors = []; trial_map_errors = []; trial_means = []
    trial_dep_loss = []; trial_marg_loss = []
    for seed in range(num_seeds):
        start = time.time()
        tf.reset_default_graph()
        tf.set_random_seed(seed)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./logs', sess.graph)
            vcs = VCSLAM(sess = sess,
                         vcs_agent = td_agent,
                         observ = zt_vals,
                         num_particles = num_train_particles,
                         num_train_steps = num_train_steps,
                         lr_d = lr_d,
                         lr_m = lr_m,
                         summary_writer = writer)

            # Train the model
            est_proposal_params, dep_loss, marg_loss = vcs.train(vcs_agent = td_agent)
            est_proposal_params = sess.run(est_proposal_params)
            est_dep_params, est_marg_params = est_proposal_params

            print("Initial marg params: {}".format(td_agent.init_marg_params().flatten()))
            print("Estimated marg params: {}".format(est_marg_params.flatten()))
            print("Initial dep params: {}".format(td_agent.init_dependency_params().flatten()))
            print("Estimated dep params: {}".format(est_dep_params.flatten()))

            # Sample the model
            particles, map_traj = vcs.sim_q(est_proposal_params,
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
        trial_dep_loss.append(np.array(dep_loss).flatten())
        trial_marg_loss.append(np.array(marg_loss).flatten())
            
        # Print elapsed time for the trial
        end = time.time()
        graph_vars = [n.name for n in tf.get_default_graph().as_graph_def().node]
        print("Trial #{}: Elapsed time {}, Graph nodes {}".format(seed,
                                                                  end - start,
                                                                  len(graph_vars)))
        #save the data
        np.savetxt('output/vcsmc_rmse_mean_{}_{}.csv'.format(num_steps,seed), trial_mean_errors, delimiter=',')
        np.savetxt('output/vcsmc_rmse_map_{}_{}.csv'.format(num_steps,seed), trial_map_errors, delimiter=',')
        np.savetxt('output/vcsmc_mean_{}_{}.csv'.format(num_steps,seed), trial_means, delimiter=',')
        np.savetxt('output/vcsmc_dep_loss_{}_{}.csv'.format(num_steps,seed), trial_dep_loss, delimiter=',')
        np.savetxt('output/vcsmc_marg_loss_{}_{}.csv'.format(num_steps,seed), trial_marg_loss, delimiter=',')


