from __future__ import print_function
import timeit
import sys
import numpy as np
import tensorflow as tf

class VCSLAMAgent():
    """
    Implements an example agent for VC-SLAM
    """
    def __init__(self,
                 num_steps=10,
                 state_dim=3,
                 num_landmarks=2,
                 landmark_dim=2,
                 latent_dim=None,
                 observ_dim=2):
        # Number of time steps
        self.num_steps = num_steps
        # Latent state dimensionality (x,y,theta)
        self.state_dim = state_dim
        # Number of landmarks
        self.num_landmarks = num_landmarks
        # Landmark dimensionality (x,y)
        self.landmark_dim = landmark_dim
        # Latent variable dimensonality
        if not latent_dim:
            self.latent_dim = self.num_steps * (self.state_dim + self.landmark_dim*self.num_landmarks)
        else:
            self.latent_dim = latent_dim
        # Observation dimensionality (range,bearing)
        self.observ_dim = observ_dim
        # Proposal parameters
        self.proposal_params = tf.placeholder(dtype=tf.float32,shape=(10,1))
        # Target model parameters
        self.target_params = tf.placeholder(dtype=tf.float32,shape=(10,1))

    # Get methods for transfering constants
    def get_num_steps(self): return self.num_steps
    def get_state_dim(self): return self.state_dim
    def get_num_landmarks(self): return self.num_landmarks
    def get_landmark_dim(self): return self.landmark_dim
    def get_latent_dim(self): return self.latent_dim
    def get_observ_dim(self): return self.observ_dim

    # TODO: decide on the structure of these params (model-specific)
    def get_dependency_param_shape(self):
        return [10,1]

    def get_marginal_param_shape(self):
        return [10,1]

    def sim_target(self):
        return tf.random_normal(shape=(self.num_steps,self.observ_dim))

    def sim_proposal(self, t, x_prev, observ, num_particles, proposal_params):
        """
        Simulate one transition by drawing a conditional sample from the
        proposal distribution: draw from every univariate marginal
        """
        return x_prev + tf.random_normal(shape=(num_particles,self.latent_dim),dtype=tf.float32)

    def log_proposal_marginal(self,t,x_curr,x_prev,obser):
        """
        Returns the log probability from the univariate marginal models
        """
        num_particles = x_curr.get_shape().as_list()[0]
        return tf.zeros(shape=(num_particles),dtype=tf.float32)

    def log_proposal_copula_sl(self,t,x_curr,x_prev,observ):
        """
        Returns the log probability from the state-landmark copula
        This function requires the multi-dimensional CDF of states,
        and the mutli-dimensional CDF of all M landmarks
        """
        # TODO: implement log of multi-dimensional state and landmark copula density
        # Need to split the tensor components into the state and land marks,
        # transform with their CDFs and
        # evaluate on the copula, and take the log
        #s_curr_tilde = self.mv_state_cdf(s_curr,x_prev,observ)

        # apply CDF for all landmarks (num_landmarks,landmark_dim)
        #l_curr_tilde = self.mv_lndmk_cdf(l_curr,x_prev,observ)

        # apply copula model
        #C_sl = self.copula_sl(s_curr_tilde,l_curr_tilde)

        # differentiate to get c?
        num_particles = x_curr.get_shape().as_list()[0]
        return tf.zeros(shape=(num_particles),dtype=tf.float32)

    def log_proposal_copula_ll(self,t,x_curr,x_prev,observ):
        """
        Log probability from the landmark-landmark copula
        """
        num_particles = x_curr.get_shape().as_list()[0]
        return tf.zeros(shape=(num_particles),dtype=tf.float32)

    def log_proposal_copula_s(self,t,x_curr,x_prev,observ):
        """
        Log probability from the state-component copula
        """
        num_particles = x_curr.get_shape().as_list()[0]
        return tf.zeros(shape=(num_particles),dtype=tf.float32)

    def log_proposal_copula_l(self,t,x_curr,x_prev,observ):
        """
        Log probability from the landmark-component copula
        """
        num_particles = x_curr.get_shape().as_list()[0]
        return tf.zeros(shape=(num_particles),dtype=tf.float32)

    def log_proposal_copula(self,t,x_curr,x_prev,observ):
        """
        Returns the log probability from the copula model described in Equation 4
        """
        return self.log_proposal_copula_sl(t,x_curr,x_prev,observ) + \
               self.log_proposal_copula_ll(t,x_curr,x_prev,observ) + \
               self.log_proposal_copula_s(t,x_curr,x_prev,observ)  + \
               self.log_proposal_copula_l(t,x_curr,x_prev,observ)

    def log_proposal(self,t,x_curr,x_prev,observ,proposal_params):
        """
        Compute the log probability using the proposal distribution described in Equation 5
        """
        # TODO: unpack proposal params and distribute to the respective models
        return self.log_proposal_copula(t,x_curr,x_prev,observ) + \
               self.log_proposal_marginal(t,x_curr,x_prev,observ)

    def log_target(self,t,x_curr,x_prev,observ):
        """
        Compute the log probability using the target distribution
        """
        num_particles = x_curr.get_shape().as_list()[0]
        return tf.zeros(shape=(num_particles),dtype=tf.float32)

    def log_weights(self,t,x_curr,x_prev,observ,proposal_params):
        """
        Compute the log weights as the difference in
        log target probabitliy and log proposal probability
        """
        return self.log_target(t,x_curr,x_prev,observ) - \
            self.log_proposal(t,x_curr,x_prev,observ,proposal_params)