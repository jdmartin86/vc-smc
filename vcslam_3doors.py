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

    def transition_model(self, x):
        dx = self.target_params[8]
        return tf.random.normal(shape=x.get_shape(), mean=x + dx, stddev=np.sqrt(self.target_params[9]))

    def measurement_model(self, x, l=[0.,2.,6.], stddev=0.1):
        c = np.random.choice([1,2,3])
        if c == 1:
            return np.abs(tf.random.normal(shape=x.get_shape(),
                                           mean=np.abs(x-l[0]),
                                           stddev=stddev)), 1
        if c == 2:
            return np.abs(tf.random.normal(shape=x.get_shape(),
                                           mean=np.abs(x-l[1]),
                                           stddev=stddev)), 2
        return np.abs(tf.random.normal(shape=x.get_shape(),
                                       mean=np.abs(x-l[2]),
                                       stddev=stddev)), 3

    def get_observations(self):
        init_mean, init_var, lm1_prior_mean, \
        lm1_prior_var, lm2_prior_mean, lm2_prior_var,\
        lm3_prior_mean, lm3_prior_var, motion_mean,\
        motion_var, meas_var = self.target_params
        state = tf.random.normal(shape=[1,1], mean=init_mean, stddev=np.sqrt(init_var))
        new_obs, landmark = self.measurement_model(state)
        states = [state]
        observations = [new_obs]
        landmarks = [landmark]
        for _ in range(self.num_steps-1):
            new_state = self.transition_model(state)
            new_obs, landmark = self.measurement_model(new_state)
            states.append(new_state)
            observations.append(new_obs)
            landmarks.append(landmark)
            state = new_state
        return observations, landmarks, states

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
        # TODO how to initialize marg params when num components > 3?
        marg_params = np.vstack([np.array([np.array([np.random.normal(loc=0.+2*t, scale=0.001),
                                                     np.random.normal(loc=2.+2*t, scale=0.001),
                                                     np.random.normal(loc=6.+2*t, scale=0.001)],
                                                    dtype=np.float32).ravel() for t in range(self.num_steps)]),# state means
                                 np.array([np.random.normal(0.,2.),# landmark #1
                                           np.random.normal(2.,2.),# landmark #2
                                           np.random.normal(6.,2.)],# landmark #3
                                          dtype=np.float32)])
        return marg_params

    def init_dependency_params(self):
        # Correlation params
        copula_params = np.array([np.array(self.cop_scale * np.random.randn(self.copula_dim)).ravel() for t in range(self.num_steps)])
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
            
            mu1 = prop_marg_params[t, 0]
            mu2 = prop_marg_params[t, 1]
            mu3 = prop_marg_params[t, 2]

            # mus = []
            # for i in range(n_components):
            #     mus.append(prop_marg_params[t, i]

            l1m = prop_marg_params[self.num_steps, 0]
            l2m = prop_marg_params[self.num_steps, 1]
            l3m = prop_marg_params[self.num_steps, 2]
            
            ps = np.array([1./3., 1./3., 1./3.], dtype=np.float32)
            x_scale = np.float32(np.sqrt(0.1))
            l_scale = np.float32(np.sqrt(0.1))

            if t > 0:
                # for i in range(n_components):
                #     mus[i] = tf.cast(mus[i] + 2., dtype=np.float32)
                mu1 = tf.cast(mu1 + 2., dtype=np.float32)
                mu2 = tf.cast(mu2 + 2., dtype=np.float32)
                mu3 = tf.cast(mu3 + 2., dtype=np.float32)            

            r_vec = prop_copula_params[t,:]
            L_mat = cc.CorrelationCholesky().forward(r_vec)

            # Defining the marginal CDFs 
            x_cdf = cg.EmpGaussianMixtureCDF(ps=ps,
                                             locs=[mu1, mu2, mu3],
                                             scales=[x_scale, x_scale, x_scale])

            l1_cdf = cg.NormalCDF(loc=l1m, scale=l_scale)
            l2_cdf = cg.NormalCDF(loc=l2m, scale=l_scale)
            l3_cdf = cg.NormalCDF(loc=l3m, scale=l_scale)

            gc = cg.WarpedGaussianCopula(
                loc=[0., 0., 0., 0.],
                scale_tril=L_mat,
                marginal_bijectors=[x_cdf, l1_cdf, l2_cdf, l3_cdf])
            #import ipdb; ipdb.set_trace()
            return gc.sample(x_prev.get_shape().as_list()[0])

    def log_proposal(self, t, x_curr, x_prev, observ, proposal_params):
        """
        Log probability from the state-component copula
        """
        prop_copula_params, prop_marg_params = proposal_params

        T = self.num_steps
        num_particles = x_prev.get_shape().as_list()[0]
        prop_copula_params = proposal_params[0]
        prop_marg_params = proposal_params[1]

        mu1 = prop_marg_params[t,0]
        mu2 = prop_marg_params[t,1]
        mu3 = prop_marg_params[t,2]

        # mus = []
        # for i in range(n_components):
        #     mus.append(prop_marg_params[t, i]

        l1m = prop_marg_params[T,0]
        l2m = prop_marg_params[T,1]
        l3m = prop_marg_params[T,2]

        # Mixture weights
        ps = np.float32([1./3., 1./3., 1./3.])
        x_scales = np.float32([np.sqrt(0.1, dtype=np.float32) for _ in range(3)])

        if t > 0:
            # for i in range(n_components):
            #     mus[i] = tf.cast(mus[i] + 2., dtype=np.float32)

            mu1 = tf.cast(mu1 + 2., dtype=np.float32)
            mu2 = tf.cast(mu2 + 2., dtype=np.float32)
            mu3 = tf.cast(mu3 + 2., dtype=np.float32)            
        locs = [mu1, mu2, mu3]
        # locs = mus
        x_cdf = cg.EmpGaussianMixtureCDF(ps=ps, locs=locs, scales=x_scales)

        # Copula params are defined over the reals, but we want a correlation matrix
        # So we use a sigmoid map to take reals to the range [-1,1]
        # r_vec = (1. - tf.exp(-prop_copula_params[t,:]))/(1. + tf.exp(-prop_copula_params[t,:])) # should be length
        # use correlationcholesky to do the right thing
        r_vec = prop_copula_params[t,:]
        L_mat = cc.CorrelationCholesky().forward(r_vec)

        # Landmark marginals
        l1_cdf = cg.NormalCDF(loc=l1m, scale=tf.sqrt(0.1))
        l2_cdf = cg.NormalCDF(loc=l2m, scale=tf.sqrt(0.1))
        l3_cdf = cg.NormalCDF(loc=l3m, scale=tf.sqrt(0.1))

        # Build a copula (can also store globally if we want) we would just
        #  have to modify self.copula.scale_tril and
        #  self.copula.marginal_bijectors in each iteration NOTE: I add
        #  tf.eye(3) to L_mat because I think the diagonal has to be > 0
        gc = cg.WarpedGaussianCopula(
            loc=[0., 0., 0., 0.],
            scale_tril=L_mat,
            marginal_bijectors=[x_cdf, l1_cdf, l2_cdf, l3_cdf])

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

    def log_target(self, t, x_curr, x_prev, observation):
        """
        This function computes the log probability of the target distribution. 
        """
        init_mean, init_var, lm1_prior_mean, \
        lm1_prior_var, lm2_prior_mean, lm2_prior_var,\
        lm3_prior_mean, lm3_prior_var, motion_mean,\
        motion_var, meas_var = self.target_params
        
        state = tf.transpose(tf.gather_nd(tf.transpose(x_prev),[[0]]))
        landmark_1 = tf.transpose(tf.gather_nd(tf.transpose(x_prev),[[1]]))
        landmark_2 = tf.transpose(tf.gather_nd(tf.transpose(x_prev),[[2]]))
        landmark_3 = tf.transpose(tf.gather_nd(tf.transpose(x_prev),[[3]]))

        new_state = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[0]]))
        new_landmark_1 = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[1]]))
        new_landmark_2 = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[2]]))
        new_landmark_3 = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[3]]))

        logF = 0.0
        logG = 0.0

        # Measurement model
        if t == 0:
            # Landmark probabilities are independent Gaussian on the current belief (Maybe update belief)?
            # log P(l_{1:3})
            logH = self.log_normal(landmark_1, lm1_prior_mean, lm1_prior_var) + \
                   self.log_normal(landmark_2, lm2_prior_mean, lm2_prior_var) + \
                   self.log_normal(landmark_3, lm3_prior_mean, lm3_prior_var)

            # log P(s0)
            logF = self.log_normal(state, init_mean, init_var)

            # log P(z0 | s0, l_{1:3})
            log_mixture_components = [tf.log(1./3.) + self.log_normal(observation[t],
                                                                      np.abs(landmark_1 - state),
                                                                      meas_var),
                                      tf.log(1./3.) + self.log_normal(observation[t],
                                                                      np.abs(landmark_2 - state),
                                                                      meas_var),
                                      tf.log(1./3.) + self.log_normal(observation[t],
                                                                      np.abs(landmark_3 - state),
                                                                      meas_var)]
            logG = tf.reduce_logsumexp(log_mixture_components, axis=0)

        else:
            # Landmark probabilities are independent Gaussian on the current belief (Maybe update belief)?
            # log P(l_{1:3})
            logH = self.log_normal(new_landmark_1, lm1_prior_mean, lm1_prior_var) + \
                   self.log_normal(new_landmark_2, lm2_prior_mean, lm2_prior_var) + \
                   self.log_normal(new_landmark_3, lm3_prior_mean, lm3_prior_var)

            # log P(s' | s)
            logF = self.log_normal(new_state,
                                   state + motion_mean,
                                   motion_var)
            
            # log P(z' | s', l_{1:3})
            log_mixture_components = [tf.log(1./3.) + self.log_normal(observation[t],
                                                                      np.abs(landmark_1 - new_state),
                                                                      meas_var),
                                      tf.log(1./3.) + self.log_normal(observation[t],
                                                                      np.abs(landmark_2 - new_state),
                                                                      meas_var),
                                      tf.log(1./3.) + self.log_normal(observation[t],
                                                                      np.abs(landmark_3 - new_state),
                                                                      meas_var)]
            logG = tf.reduce_logsumexp(log_mixture_components, axis=0)

        return logF + logG + logH

    def log_weights(self, t, x_curr, x_prev, observ, proposal_params):
        target_log = self.log_target(t, x_curr, x_prev, observ)
        prop_log = self.log_proposal(t, x_curr, x_prev, observ, proposal_params)
        return target_log - prop_log

if __name__ == '__main__':

    # Number of steps for the trajectory
    num_steps = 2 # Must be greater than 1
    # Number of particles to use during training
    num_train_particles = 10
    # Number of iterations to fit the proposal parameters
    num_train_steps = 0
    # Learning rate for the marginal
    lr_m = 1.e-2
    # Learning rate for the copula
    lr_d = 1.e-2
    # Number of random seeds for experimental trials
    num_seeds = 1
    # Proposal initial scale
    prop_scale = 2.0
    # Copula initial scale
    cop_scale = 0.1

    # True target parameters
    lm1_prior_mean = np.array([[0.]], dtype=np.float32)
    lm2_prior_mean = np.array([[2.]], dtype=np.float32)
    lm3_prior_mean = np.array([[6.]], dtype=np.float32)
    lm1_prior_var = np.array([[0.1]], dtype=np.float32)
    lm2_prior_var = np.array([[0.1]], dtype=np.float32)
    lm3_prior_var = np.array([[0.1]], dtype=np.float32)
    motion_mean = np.array([[2.]], dtype=np.float32)
    motion_var = np.array([[0.1]], dtype=np.float32)
    meas_var = np.array([[0.1]], dtype=np.float32)
    init_mean = np.array([[0.]], dtype=np.float32)
    init_var = np.array([[0.1]], dtype=np.float32)
    target_params = [init_mean, init_var,
                     lm1_prior_mean, lm1_prior_var,
                     lm2_prior_mean, lm2_prior_var,
                     lm3_prior_mean, lm3_prior_var,
                     motion_mean, motion_var,
                     meas_var]

    # Create the agent
    td_agent = ThreeDoorsAgent(target_params=target_params,
                               num_steps=num_steps,
                               prop_scale=prop_scale,
                               cop_scale=cop_scale)

    # Generate observations
    observ, landmarks, states = td_agent.get_observations()
    with tf.Session() as sess:
        observ, states = sess.run([observ, states])
    states = np.array(np.squeeze(states))

    # Reference observations
    if True:
        states = np.array([-0.4351809,  1.6902565,  4.002844],
                          dtype=np.float32)
        landmarks = [1,2,1]
        o1 = np.array([[0.29190797]], dtype=np.float32)
        o2 = np.array([[0.4014704]], dtype=np.float32)
        o3 = np.array([[4.1063337]], dtype=np.float32)
        observ = [o1,o2,o3]

    print("States: {}".format(states))
    print("Observations: {}".format(observ))
    print("Landmarks: {}".format(landmarks))

    np.savetxt('output/observations.csv',
               np.array([o.flatten() for _,o in enumerate(observ)]))
    np.savetxt('output/states.csv',
               states,
               delimiter=',')

    truth = []
    for _, s in enumerate(states):
        truth.append(np.array([s, 0, 2, 6], dtype=np.float32))
    truth = np.array(truth)               
    np.savetxt('output/trajectory_ref.csv', truth, delimiter=',')

    trial_mean_errors = []; trial_map_errors = []; trial_means = []
    trial_dep_loss = []; trial_marg_loss = []
    for seed in range(num_seeds):
        start = time.time()
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./logs', sess.graph)
            vcs = VCSLAM(sess = sess,
                         vcs_agent = td_agent,
                         observ = observ,
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
            particles, map_traj, logw = vcs.sim_q(est_proposal_params,
                                                  target_params,
                                                  observ,
                                                  td_agent,
                                                  num_samples=num_train_particles)
            particles, map_traj, logw = sess.run([particles, map_traj, logw])
        particles = np.squeeze(np.array(particles))
        print('Estimated trajectory samples: {}'.format(particles))

        # record trial data
        mean_traj = np.mean(particles[num_steps-1,:,:], 0)
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

        print("lowg: {}".format(logw))
        print("Probs: {}".format(np.exp(logw)/np.sum(np.exp(logw))))
        np.savetxt('output/vcsmc_weights.csv', np.exp(logw)/np.sum(np.exp(logw)), delimiter=',')
        for t in range(num_steps):
            np.savetxt('output/vcsmc_particles_{}.csv'.format(t),
                       particles[t,:,:],
                       delimiter=',')

