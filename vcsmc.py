from __future__ import print_function
import timeit
import sys
import numpy as np
import tensorflow as tf

class VCSLAM():
    """
    This class implements the VC-SLAM training procedure and
    a sampling method for the learned proposal.
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
        # log_weights: A Tensor of shape (num_particles, num_steps)
        #  containing the log weights at each timestep of the particle filter.
        # self.logw = tf.random_uniform((self.num_particles,self.num_steps),min=0.0,max=1.0)

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
        resampling_dist = tf.contrib.distributions.Categorical(logits=log_weights)
        return resampling_dist.sample(sample_shape=(num_samples))

    def vsmc_lower_bound(self, vcs_agent, proposal_params):
        """
        Estimate the VSMC lower bound. Amenable to (biased) reparameterization
        gradients.

        Inputs:

        .. math::
            ELBO(\theta,\lambda) =
            \mathbb{E}_{\phi}\left[\nabla_\lambda \log \hat p(y_{1:T}) \right]

        """
        # Initialize SMC
        x_curr = tf.zeros(dtype=tf.float32,shape=(self.num_particles,self.latent_dim))
        x_prev = tf.zeros(dtype=tf.float32,shape=(self.num_particles,self.latent_dim))

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
            else:
                x_prev = x_curr

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
            logw_tilde = tf.debugging.check_numerics(logw_tilde, "Error, ahhh!!!")
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

    def sim_q(self, prop_params, model_params, y, vcs_obj, num_samples=1, num_particles=None):
        """
        Simulates a single sample from the VSMC approximation.
        This returns the SLAM solution
        This procedure is the same as the objective, but it saves the trajectory
        """

        # Allow user to simulate q with more particles than we trained with
        if not num_particles:
            num_particles = self.num_particles

        # Initialize SMC
        x_curr = tf.zeros(dtype=tf.float32,shape=(self.num_steps,num_particles,self.latent_dim))
        x_prev = tf.zeros(dtype=tf.float32,shape=(num_particles,self.latent_dim))

        # Unnormalized particle weights
        logw_tilde = tf.zeros(dtype=tf.float32,shape=(num_particles))
        logZ = tf.zeros(dtype=tf.float32,shape=(1))

        X = tf.zeros(dtype=tf.float32, shape=(self.num_steps, num_particles, self.latent_dim))

        # For effective sample size (ESS) calculations
        # TODO: implement after testing regular resampling (04/22)
        #w      = tf.nn.softmax(logits=logW)
        #ESS = 1./np.sum(W**2)/N

        # start_smc = timeit.default_timer()
        for t in range(self.num_steps):
            # Resampling
            # Shape of x_prev (num_particles,latent_dim)
            if t > 0:
                ancestors = self.resampling(logw_tilde, num_particles)
                x_prev = tf.gather(x_curr,ancestors,axis=0) #TODO: this indexing won't work - just for prototyping
            else:
                x_prev = x_curr[0,:,:]

            # Propagation
            # This simulates one transition from the proposal distribution
            # Shape of x_curr (num_particles,latent_dim)
            # TODO: Revisit the arguments when you implement the class with the proposal and couplas
            x_curr = vcs_obj.sim_proposal(t, x_prev, self.observ, prop_params)

            # Weighting
            # Get the log weights for the current timestep
            # Shape of logw_tilde (num_particles)
            logw_tilde = vcs_obj.log_weights(t, x_curr, x_prev, self.observ, prop_params)
            # print(logw_tilde)
            max_logw_tilde = tf.math.reduce_max(logw_tilde)
            logw_tilde_adj = logw_tilde - max_logw_tilde
            logZ += tf.math.reduce_logsumexp(logw_tilde_adj) - tf.log(tf.to_float(num_particles)) + max_logw_tilde

            # Not sure if this is correct at all - Kevin
            W = tf.exp(logw_tilde_adj)
            W /= tf.reduce_sum(W)
            logW = tf.log(W)

            #w = tf.nn.softmax(logits=logw_tilde_adj)
            #ESS = 1./tf.reduce_sum(w**2)/self.num_particles

        Bs = self.sample_traj(logw_tilde, num_samples)
        return tf.gather(x_curr,Bs,axis=0)

    def train(self,vcs_agent):
        """
        Creates the training operation
        """
        print("Starting training")
        dependency_initializer = tf.constant_initializer(vcs_agent.init_dependency_params())
        marginal_initializer = tf.constant_initializer(vcs_agent.init_marg_params())

        # Initialize the parameters
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
            proposal_params = [dependency_params,marginal_params]

            # Compute losses and define the learning procedures
            loss = -self.vsmc_lower_bound(vcs_agent,proposal_params)
            loss_summary = tf.summary.scalar(name='elbo', tensor=tf.squeeze(-loss))
            summary_op = tf.summary.merge_all()

            learn_dependency = self.optimizer(learning_rate=self.lr_d).minimize(loss, var_list=dependency_params)
            learn_marginal   = self.optimizer(learning_rate=self.lr_m).minimize(loss, var_list=marginal_params)

            # Start the session
            self.sess.run(tf.global_variables_initializer())

            # Top-level training loop
            # TODO: add logging for loss terms
            iter_display = 100
            print("    Iter    |    ELBO    ")
            # Expectation Maximization loop
            for i in range(self.num_train_steps):

                # Train the dependency model
                for it in range(self.num_dependency_train_steps):
                    _, loss_curr = self.sess.run([learn_dependency, loss])

                    if np.isnan(loss_curr):
                        print("NAN loss:", i, it)
                        return None

                    # dep_losses[it] = loss_curr

                # Train the marginal model
                for it in range(self.num_marginal_train_steps):
                    _, loss_curr, summary_str = self.sess.run([learn_marginal, loss, summary_op])

                    if np.isnan(loss_curr):
                        print("NAN loss:", i, it)
                        return None

                    # mar_losses[it] = loss_curr

                #self.summary_writer.add_summary(summary_str, it)
                if i % iter_display == 0:
                    message = "{:15}|{!s:20}".format(i, -loss_curr)
                    print(message)

        return proposal_params
