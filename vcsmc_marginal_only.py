from __future__ import print_function
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

class VCSLAM():
    """
    This class implements the VC-SLAM training procedure and
    a sampling method for the learned proposal.
    """
    def __init__(self,
                vcs_agent,
                observ,
                num_particles,
                num_train_steps = 1000,
                lr_d = 0.0001,
                lr_m = 0.0001,
                adapt_resamp = False,
                summary_writer=None,
                summary_writing_frequency=500,
                rs=np.random.RandomState(0)):
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
        # Dependency model learning rate
        self.lr_d = lr_d
        # Marginal model learning rate
        self.lr_m = lr_m
        # Cached constant: the logarithm of the number of particles
        self.log_num_particles = tf.log(tf.to_float(self.num_particles))

        self.rs = rs

    def resampling(self, log_weights):
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
        resampling_dist = tf.contrib.distributions.Categorical(logits=log_weights)
        ancestors = tf.stop_gradient(
            resampling_dist.sample(sample_shape=(self.num_particles),seed=self.rs.randint(0,1234)))
        return ancestors

    def sample_traj(self, log_weights):
        """
        Draw index from the particle set
        Args:
            log_weights: log importance weights
        Returns:
            index: An ancenstral index
        """
        # print("shape of log weights? ", log_weights)
        resampling_dist = tf.contrib.distributions.Categorical(logits=log_weights)
        # print(log_weights)
        samp = resampling_dist.sample(seed=self.rs.randint(0,1234))
        return samp

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
        return logZ

    def sim_q(self, prop_params, model_params, y, vcs_obj):
        """
        Simulates a single sample from the VSMC approximation.
        This returns the SLAM solution
        This procedure is the same as the objective, but it saves the trajectory
        """

        # Initialize SMC
        x_curr = tf.zeros(dtype=tf.float32,shape=(self.num_steps,self.num_particles,self.latent_dim))
        x_prev = tf.zeros(dtype=tf.float32,shape=(self.num_particles,self.latent_dim))

        # Unnormalized particle weights
        logw_tilde = tf.zeros(dtype=tf.float32,shape=(self.num_particles))
        logZ = tf.zeros(dtype=tf.float32,shape=(1))

        X = tf.zeros(dtype=tf.float32, shape=(self.num_steps, self.num_particles, self.latent_dim))

        # For effective sample size (ESS) calculations
        # TODO: implement after testing regular resampling (04/22)
        #w      = tf.nn.softmax(logits=logW)
        #ESS = 1./np.sum(W**2)/N

        for t in range(self.num_steps):
            # Resampling
            # Shape of x_prev (num_particles,latent_dim)
            if t > 0:
                ancestors = self.resampling(logw_tilde)
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
            logZ += tf.math.reduce_logsumexp(logw_tilde_adj) - tf.log(tf.to_float(self.num_particles)) + max_logw_tilde

            # Not sure if this is correct at all - Kevin
            W = tf.exp(logw_tilde_adj)
            W /= tf.reduce_sum(W)
            logW = tf.log(W)

            #w = tf.nn.softmax(logits=logw_tilde_adj)
            #ESS = 1./tf.reduce_sum(w**2)/self.num_particles

        # Sample from the empirical approximation
        # print("LogW tilde: ", logw_tilde)
        B = self.sample_traj(logw_tilde)
        # B = self.sample_traj(logW)
        # print("B: ", B)
        return tf.gather(x_curr,B,axis=0)
        # return x_curr, B

    def train(self,vcs_agent):
        """
        Creates the top-level computation graph for training
        """
        print("Starting training")
        # tf.reset_default_graph()
        # initializer = tf.contrib.layers.xavier_initializer()
        dependency_initializer = tf.contrib.layers.xavier_initializer()
        marginal_initializer = tf.constant_initializer()

        # Initialize the parameters
        if vcs_agent.get_dependency_param_shape() == 0:
            dependency_params = []
        else:
            dependency_params = tf.get_variable( "theta",
                                                dtype=tf.float32,
                                                shape=vcs_agent.get_dependency_param_shape(),
                                                initializer=dependency_initializer)
        marginal_params   = tf.get_variable( "eta",
                                            dtype=tf.float32,
                                            shape=vcs_agent.get_marginal_param_shape(),
                                            initializer=marginal_initializer)
        print("Marginal params shape", marginal_params.shape)
        proposal_params = [dependency_params,marginal_params]
        print("Length of proposal params", len(proposal_params))
        # Compute losses and define the learning procedures
        loss = -self.vsmc_lower_bound(vcs_agent,proposal_params)

        # learn_dependency = self.optimizer(learning_rate=self.lr_d).minimize(loss, var_list=dependency_params)
        learn_marginal   = self.optimizer(learning_rate=self.lr_m).minimize(loss, var_list=marginal_params)

        # Start the session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print("Original marginal_params:\n", marginal_params.eval(session=sess))

        # Top-level training loop
        # TODO: add logging for loss terms
        iter_display = 100
        print("    Iter    |    ELBO    ")
        for it in range(self.num_train_steps):

            # Train the dependency model
            # _, loss_curr = sess.run([learn_dependency, loss])

            # if np.isnan(loss_curr):
            #     print("NAN loss:", it)
            #     break

            # dep_losses[it] = loss_curr

            # Train the marginal model
            _, loss_curr = sess.run([learn_marginal, loss])

            if np.isnan(loss_curr):
                print("NAN loss:", it)
                # Break everything if the loss goes to NaN
                return None
                break

            # mar_losses[it] = loss_curr

            if it % iter_display == 0:
                message = "{:15}|{:20}".format(it, -loss_curr)
                print(message)
        print("Final marginal params:\n", marginal_params.eval(session=sess))
        return proposal_params, sess


if __name__ == '__main__':
    # Instantiate the VC-SLAM agent, which includes a target and a proposal dist
    # TODO: Implement this class with the test problem we consider, along with associated methods
    # vcs_agent = VCSLAMAgent()
    vcs_agent = VCSLAMAgent(num_steps=1, state_dim=1, num_landmarks=2, landmark_dim=1, latent_dim=3, observ_dim=1)

    # Simulate the system to obtain a sequence of observations
    observ = vcs_agent.sim_target()

    # Instantiate the VC-SLAM procedure
    vcs = VCSLAM(vcs_agent = vcs_agent , observ = observ , num_particles = 10)

    # Train the approximate model
    vcs.train(vcs_agent = vcs_agent)

    # Output the SLAM solution
    #vcs_solution = vcs.sim_q()
