from __future__ import print_function
import timeit
import sys
import numpy as np
import tensorflow as tf

class VCSLAM():
    """
    This class implements the VC-SLAM training procedure and a sampling method
    for the learned proposal.
    """
    def __init__(self,
                sess,
                vcs_agent,
                observ,
                num_particles,
                num_train_steps = 1000,
                num_dependency_train_steps = 1,
                num_marginal_train_steps = 1,
                lr_d = 0.0001,
                lr_m = 0.0001,
                adapt_resamp = False,
                summary_writer=None,
                summary_writing_frequency=500):
        # TF Session
        self.sess = sess
        # VC-SLAM agent
        self.vcs_agent = vcs_agent
        # Number of time steps on the horizon
        self.num_steps = vcs_agent.get_num_steps()
        # State dimensionality
        self.state_dim = vcs_agent.get_state_dim()
        # Number of landmarks
        self.num_landmarks = vcs_agent.get_num_landmarks()
        # Dimensonality of landmark variables
        self.landmark_dim = vcs_agent.get_landmark_dim()
        # Dimensionality of the latent space
        self.latent_dim = vcs_agent.get_latent_dim()
        # Dimensionality of the observable space
        self.observ_dim = vcs_agent.get_observ_dim()
        # Observation sequence (num_steps,observ_dim) tensor
        self.observ = observ
        # Number of trajectory samples
        self.num_particles = num_particles
        # Boolean flag to invoke adaptive resampling
        self.adapt_resamp = adapt_resamp
        # Training procedure is Adam
        self.optimizer = tf.train.AdamOptimizer
        # Number of training iterations
        self.num_train_steps = num_train_steps
        # Number of iterations to fit the dependency parameters
        self.num_dependency_train_steps = num_dependency_train_steps
        # Number of iterations to fit the marginal parameters
        self.num_marginal_train_steps = num_marginal_train_steps
        # Dependency model learning rate
        self.lr_d = lr_d
        # Marginal model learning rate
        self.lr_m = lr_m
        # Cached constant: the logarithm of the number of particles
        self.log_num_particles = tf.log(tf.to_float(self.num_particles))
        # Tensorboard summary writer and log frequency
        self.summary_writer = summary_writer
        self.summary_writing_frequency = summary_writing_frequency

    def resampling(self, log_weights, num_particles=None):
        """
        Stratified resampling
        Args:
            log_weights: log importance weights
        Returns:
            ancestors: A tensor of ancenstral indices (num_particles,1)
        """
        # Calculate the ancenstral indices via resampling. Because we maintain the
        # log unnormalized weights, we pass the weights in as logits, allowing
        # the distribution object to apply a softmax and normalize them.
        # log_weights = tf.gather(self.logw,t,axis=1)
        if not num_particles:
            num_particles = self.num_particles
        resampling_dist = tf.contrib.distributions.Categorical(logits=log_weights)
        ancestors = tf.stop_gradient(
            resampling_dist.sample(sample_shape=(num_particles)))
        return ancestors
    
    def sample_traj(self, log_weights, num_samples=1):
        """
        Draw index from the particle set
        Args:
            log_weights: log importance weights
        Returns:
            index: An ancenstral index
        """
        resampling_dist = tf.contrib.distributions.Categorical(
            logits=log_weights)
        return tf.to_int64(resampling_dist.sample(sample_shape=num_samples))

    def vsmc_lower_bound(self, vcs_agent, proposal_params):
        """
        Estimate the VSMC lower bound. Amenable to (biased) reparameterization
        gradients.

        Inputs:

        .. math::
            ELBO(\theta,\lambda) =
            \mathbb{E}_{\phi}\left[\nabla_\lambda \log \hat p(y_{1:T}) \right]

        """
        # Initialize SMC (POOR INIT! SHOULD BE SAMPLING FROM THE PRIOR)
        x_curr = tf.zeros(dtype=tf.float32,shape=(self.num_particles,self.latent_dim))
        x_prev = vcs_agent.sim_proposal(0,tf.zeros(dtype=tf.float32,shape=(self.num_particles,self.latent_dim)), self.observ, proposal_params)

        # Unnormalized particle weights
        logw_tilde = tf.zeros(dtype=tf.float32,shape=(self.num_particles))
        logZ = tf.zeros(dtype=tf.float32,shape=(1))

        # For effective sample size (ESS) calculations
        # TODO: implement after testing regular resampling (04/22)
        #w      = tf.nn.softmax(logits=logW)
        #ESS = 1./np.sum(W**2)/N

        # start_smc = timeit.default_timer()
        for t in range(self.num_steps):
            # Resampling
            # Shape of x_prev (num_particles,latent_dim)
            # Not doing adaptive resampling
            if t > 0:
                ancestors = self.resampling(logw_tilde)
                x_prev = tf.gather(x_curr,ancestors,axis=0)

            # Propagation
            # This simulates one transition from the proposal distribution
            # Shape of x_curr (num_particles,latent_dim)
            # TODO: Revisit the arguments when you implement the class with the proposal and couplas
            x_curr = vcs_agent.sim_proposal(t, x_prev, self.observ, proposal_params)
            # x_curr = vcs_agent.sim_proposal(t, x_prev, self.observ, self.num_particles, proposal_params)

            # Weighting
            # Get the log weights for the current timestep
            # Shape of logw_tilde (num_particles)
            logw_tilde = vcs_agent.log_weights(t, x_curr, x_prev, self.observ, proposal_params)
            max_logw_tilde = tf.reduce_max(logw_tilde)
            logw_tilde_adj = logw_tilde - max_logw_tilde

            # Temporarily switched self.log_num_particles to its definition
            # i.e. tf.log(tf.to_float(self.num_particles))
            # This fixed a graph error
            logZ += max_logw_tilde + tf.reduce_logsumexp(logw_tilde_adj) - tf.log(tf.to_float(self.num_particles))

            #w = tf.nn.softmax(logits=logw_tilde_adj)
            #ESS = 1./tf.reduce_sum(w**2)/self.num_particles
        # print("Train SMC Time: ", (timeit.default_timer() - start_smc))
        return logZ

    def get_traj_samples(self, trajectories, logits, num_samples):
      step_indices = tf.range(tf.to_int64(self.num_steps))[:, None]
      traj_indices = self.sample_traj(logits, num_samples)

      step_indices_tiled = tf.tile(step_indices, [1, num_samples])
      traj_indices_tiled = tf.tile(traj_indices[None,:], [self.num_steps, 1])
      step_indices_tiled = tf.expand_dims(step_indices_tiled, 2)
      traj_indices_tiled = tf.expand_dims(traj_indices_tiled, 2)

      indices = tf.concat([step_indices_tiled, traj_indices_tiled], 2)

      return tf.gather_nd(trajectories, indices)

    def get_map_traj(self, trajectories, logits):
      # Compute the index of highest probability trajectory.
      dist = tf.contrib.distributions.Categorical(logits=logits)
      traj_index = tf.expand_dims(tf.argmax(dist.probs), 0)

      step_indices = tf.range(tf.to_int64(self.num_steps))[:, None]
      traj_index_tiled = tf.tile(traj_index, [self.num_steps])[:, None]
      indices = tf.concat([step_indices, traj_index_tiled], 1)

      return tf.gather_nd(trajectories, indices)

    def sim_q(self,
              prop_params,
              model_params,
              y,
              vcs_obj,
              num_samples=1):
      """
      Simulates a single sample from the VSMC approximation.
      This returns the SLAM solution
      This procedure is the same as the objective, but it saves the trajectory
      """
      # Initialize SMC
      x_curr = tf.zeros([self.num_particles, self.latent_dim])
      #x_prev = 
      x_prev = vcs_obj.sim_proposal(0, tf.zeros([self.num_particles, self.latent_dim], dtype=tf.float32), self.observ, prop_params)
      
      # Unnormalized particle weights
      logw_tilde = tf.zeros(self.num_particles)

      # Effective sample size.
      #w      = tf.nn.softmax(logits=logW)
      #ESS = 1./np.sum(W**2)/N

      trajectories = []
      for t in range(self.num_steps):
        # Resampling
        # Shape of x_prev (num_particles,latent_dim)
        if t > 0:
          ancestors = self.resampling(logw_tilde, self.num_particles)
          x_prev = tf.gather(x_curr,ancestors,axis=0)

        # Propagation
        # This simulates one transition from the proposal distribution
        # Shape of x_curr: num_particles x latent_dim
        x_curr = vcs_obj.sim_proposal(t, x_prev, self.observ, prop_params)
        
        # Weighting
        # Get the log weights for the current timestep
        # Shape of logw_tilde (num_particles)
        logw_tilde = vcs_obj.log_weights(t, x_curr, x_prev, self.observ,
                                         prop_params)

        # Effective sample size.
        # TODO enable after testing resample at each step
        #max_logw_tilde = tf.math.reduce_max(logw_tilde)
        #logw_tilde_adj = logw_tilde - max_logw_tilde
        #w = tf.nn.softmax(logits=logw_tilde_adj)
        #ESS = 1./tf.reduce_sum(w**2)/self.num_particles

        # Append trajectory elements
        trajectories.append(x_prev)

        #with tf.Session() as sess:
        #    print(sess.run([x_prev]))
        #    print(sess.run([logw_tilde]))
      # Stack trajectories into a tensor.
      trajectories = tf.stack(trajectories)

      # Return a batch of sampled trajectories and the MAP trajectory.
      #chosen_trajs = self.get_traj_samples(trajectories, logw_tilde, num_samples)
      map_traj = self.get_map_traj(trajectories, logw_tilde)
      return trajectories, map_traj, logw_tilde

    def train(self,vcs_agent, verbose=True):
        """
        Creates the training operation
        """
        dependency_initializer = tf.constant_initializer(vcs_agent.init_dependency_params())
        marginal_initializer = tf.constant_initializer(vcs_agent.init_marg_params())

        # Initialize the parameters
        # Can we just use place holders for the params?
        with tf.variable_scope("vcsmc", reuse=tf.AUTO_REUSE):
            dependency_params = tf.get_variable( "theta",
                                                 dtype=tf.float32,
                                                 shape=vcs_agent.get_dependency_param_shape(),
                                                 initializer=dependency_initializer)
            marginal_params   = tf.get_variable( "eta",
                                                dtype=tf.float32,
                                                shape=vcs_agent.get_marginal_param_shape(),
                                                initializer=marginal_initializer)
            #print("Marginal params shape", marginal_params.shape)
            proposal_params = [dependency_params, marginal_params]

            # Compute losses and define the learning procedures
            loss = -self.vsmc_lower_bound(vcs_agent, proposal_params)

            learn_dependency = self.optimizer(learning_rate=self.lr_d).minimize(loss, var_list=dependency_params)
            learn_marginal = self.optimizer(learning_rate=self.lr_m).minimize(loss, var_list=marginal_params)

            # Start the session
            self.sess.run(tf.global_variables_initializer())

            # Top-level training loop
            # TODO: add logging for loss terms
            iter_display = 100
            print("    Iter    |    ELBO    ")
            # Expectation Maximization loop
            marginal_loss = []; dependency_loss = []
            for i in range(self.num_train_steps):

                # Train the dependency model
                for it in range(self.num_dependency_train_steps):
                    _, loss_curr = self.sess.run([learn_dependency, loss])
                    dependency_loss.append(loss_curr)
                    if np.isnan(loss_curr):
                        print("NAN loss:", i, it)
                        return None

                # Train the marginal model
                for it in range(self.num_marginal_train_steps):
                    _, loss_curr = self.sess.run([learn_marginal, loss])
                    marginal_loss.append(loss_curr)
                    if np.isnan(loss_curr):
                        print("NAN loss:", i, it)
                        return None

                #self.summary_writer.add_summary(summary_str, it)
                if verbose and i % iter_display == 0:
                    message = "{:15}|{!s:20}".format(i, loss_curr)
                    print(message)

        return proposal_params, dependency_loss, marginal_loss
