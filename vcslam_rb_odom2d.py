import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow.contrib.distributions as tfd

import plotting

from vcsmc import *
import vcslam_agent

import copula_gaussian as cg

# Remove warnings
tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()

class RangeBearingAgent(vcslam_agent.VCSLAMAgent):
    def __init__(self,
                 target_params,
                 num_steps=2,
                 state_dim=3,
                 num_landmarks=0,
                 landmark_dim=2,
                 latent_dim=None,
                 observ_dim=2,
                 rs=np.random.RandomState(0),
                 prop_scale=0.5,
                 cop_scale=1.0):
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
        # if not latent_dim:
        #     self.latent_dim = self.num_steps * (self.state_dim + self.landmark_dim*self.num_landmarks)
        # else:
        #     self.latent_dim = latent_dim
        self.latent_dim = (self.state_dim + self.landmark_dim*self.num_landmarks)
        # Observation dimensionality (direct observations of x_door)
        self.observ_dim = observ_dim
        # Proposal params
        self.proposal_params = tf.placeholder(dtype=tf.float32,shape=(10,1))

        self.prop_scale = prop_scale
        self.cop_scale = cop_scale
        # Random state for sampling
        self.rs = rs

        # initialize dependency models
        self.init_dependency_params()

    def transition_model(self, x):
        init_pose, init_cov, A, Q, C, R = self.target_params
        return tf.matmul(A, x)

    def measurement_model(self, x):
        init_pose, init_cov, A, Q, C, R = self.target_params
        return tf.matmul(C, x)

    def get_dependency_param_shape(self):
        return [self.num_steps, self.state_dim*2]

    def get_marginal_param_shape(self):
        return [self.num_steps, self.state_dim*3]

    def init_marg_params(self):
        T = self.num_steps
        Dx = self.state_dim
        marg_params = np.array([np.array([self.prop_scale * self.rs.randn(Dx), # Bias
                 1. + self.prop_scale * self.rs.randn(Dx), # Linear times A/mu0
                 self.prop_scale * self.rs.randn(Dx)]).ravel() # Log-var
                for t in range(T)])
        return marg_params

    def init_dependency_params(self):
        # State-component copula model represents a joint dependency distribution
        # over the state components
        mean = tf.zeros(shape=self.state_dim, dtype=tf.float32)
        scale_tril = tf.eye(self.state_dim, dtype=tf.float32) 
        self.copula_s = cg.WarpedGaussianCopula(
            loc=mean,
            scale_tril=scale_tril,
            marginal_bijectors=[
                tfd.Normal(loc=0., scale=1.),
                tfd.Normal(loc=0., scale=1.),
                tfd.Normal(loc=0., scale=1.)])

        T = self.num_steps
        Dx = self.state_dim
        # Each correlation param should be in [-1,1]
        # So we compute the actual correlation param as
        # (1 - exp(-x)) / (1 + exp(-x)) which you can verify as 2*sigmoid(x) - 1
        # where sigmoid(x): R -> [0,1] = 1 / (1 + exp(-x))
        # and we just rescale that to be [-1,1]
        # I think there's a better way I just need to think about it for a bit - Kevin
        copula_params = np.array([np.array(self.cop_scale * self.rs.randn(Dx*2)).ravel() # correlation/covariance params
                                  for t in range(T)])
        return np.array([])
    def generate_data(self):
        # print(self.target_params)
        init_pose, init_cov, A, Q, C, R = self.target_params
        Dx = init_pose.get_shape().as_list()[0]
        Dz = R.get_shape().as_list()[0]

        x_true = []
        z_true = []

        for t in range(self.num_steps):
            if t > 0:
                x_true.append(tf.transpose(tfd.MultivariateNormalFullCovariance(loc=tf.transpose(self.transition_model(x_true[t-1])),
                                                                   covariance_matrix=Q).sample(seed=self.rs.randint(0,1234))))
            else:
                x_sample = tf.transpose(tfd.MultivariateNormalFullCovariance(loc=tf.transpose(init_pose),
                                                                   covariance_matrix=init_cov).sample(seed=self.rs.randint(0,1234)))
                x_sample = init_pose
                x_true.append(x_sample)
            z_true.append(tf.transpose(tfd.MultivariateNormalFullCovariance(loc=tf.transpose(self.measurement_model(x_true[t])),
                                                               covariance_matrix=R).sample(seed=self.rs.randint(0,1234))))
        return x_true, z_true

    def lgss_posterior_params(self, observ, T):
        """
            Apply a Kalman filter to the linear Gaussian state space model
            Returns p(x_T | z_{1:T}) when supplied with z's and T
        """
        init_pose, init_cov, A, Q, C, R = self.target_params
        Dx = init_pose.get_shape().as_list()[0]
        Dy = R.get_shape().as_list()[0]
        log_likelihood = 0.0
        xfilt = tf.zeros(Dx)
        Pfilt = tf.zeros([Dx, Dx])
        xpred = init_pose
        Ppred = init_cov
        for t in range(self.num_steps):
            if t > 0:
                # Predict Step
                xpred = self.transition_model(xfilt)
                Ppred = tf.matmul(A, tf.matmul(Pfilt, tf.transpose(A))) + Q
            # Update step
            yt = observ[t] - self.measurement_model(xpred)
            S = tf.matmul(C, tf.matmul(Ppred, tf.transpose(C))) + R
            K = tf.transpose(tf.linalg.solve(S, tf.matmul(C, Ppred)))
            xfilt = xpred + tf.matmul(K,yt)
            Pfilt = Ppred - tf.matmul(K, tf.matmul(C,Ppred))
        return xfilt, Pfilt

    def sim_proposal(self, t, x_prev, observ, proposal_params):
        init_pose, init_cov, A, Q, C, R = self.target_params
        num_particles = x_prev.get_shape().as_list()[0]
        proposal_marg_params = proposal_params[1]
        mut = proposal_marg_params[t,0:3]
        lint = proposal_marg_params[t,3:6]
        log_s2t = proposal_marg_params[t,6:9]
        s2t = tf.exp(log_s2t)
        if t > 0:
            mu = mut + tf.transpose(self.transition_model(tf.transpose(x_prev)))*lint
        else:
            mu = mut + lint*tf.reshape(init_pose, (self.state_dim,))
        sample = mu + tf.random.normal(x_prev.get_shape().as_list(),seed=self.rs.randint(0,1234))*tf.sqrt(s2t)
        return sample

    def log_proposal_copula_sl(self,t,x_curr,x_prev,observ,prop_copula_params):
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
        # xx_dist = NormalCDF()
        # x_tilde =
        return tf.zeros(shape=(num_particles),dtype=tf.float32)

    def log_proposal_copula_ll(self,t,x_curr,x_prev,observ,prop_copula_params):
        """
        Log probability from the landmark-landmark copula
        """
        num_particles = x_curr.get_shape().as_list()[0]
        return tf.zeros(shape=(num_particles),dtype=tf.float32)

    def log_proposal_copula_s(self,t,x_curr,x_prev,observ,prop_copula_params):
        """
        Log probability from the state-component copula
        """
        # TODO: implement Gaussian copula here
        r_vec = prop_copula_params[t,:] # should be length 6

        # TODO: relocate to the init marginal params routine
        R_mat = tfd.fill_triangular(r_vec)
        num_particles = x_curr.get_shape().as_list()[0]
        x1_emp_mean = tf.reduce_mean(x_curr[:,0])
        x2_emp_mean = tf.reduce_mean(x_curr[:,1])
        x3_emp_mean = tf.reduce_mean(x_curr[:,2])
        x1_emp_std = tf.math.reduce_std(x_curr[:,0])
        x2_emp_std = tf.math.reduce_std(x_curr[:,1])
        x3_emp_std = tf.math.reduce_std(x_curr[:,2])
        x1_cdf = cg.NormalCDF(loc=x1_emp_mean, scale=x1_emp_std)
        x2_cdf = cg.NormalCDF(loc=x2_emp_mean, scale=x2_emp_std)
        x3_cdf = cg.NormalCDF(loc=x3_emp_mean, scale=x3_emp_std)


        u1 = x1_cdf._forward(x_curr[:,0])
        u2 = x2_cdf._forward(x_curr[:,1])
        u3 = x3_cdf._forward(x_curr[:,2])
        uni_cdf = cg.NormalCDF()
        c1 = uni_cdf._inverse(u1)
        c2 = uni_cdf._inverse(u2)
        c3 = uni_cdf._inverse(u3)

        #mapping = lambda x: tf.log(self.copula_s.prob(x))
        #print(tf.map_fn(mapping, x_curr))
        return tf.zeros(shape=(num_particles),dtype=tf.float32)

    def log_proposal_copula_l(self,t,x_curr,x_prev,observ,prop_copula_params):
        """
        Log probability from the landmark-component copula
        """
        num_particles = x_curr.get_shape().as_list()[0]
        return tf.zeros(shape=(num_particles),dtype=tf.float32)

    def log_proposal_copula(self,t,x_curr,x_prev,observ,prop_copula_params):
        """
        Returns the log probability from the copula model described in Equation 4
        """
        return self.log_proposal_copula_sl(t,x_curr,x_prev,observ,prop_copula_params) + \
               self.log_proposal_copula_ll(t,x_curr,x_prev,observ,prop_copula_params) + \
               self.log_proposal_copula_s(t,x_curr,x_prev,observ,prop_copula_params)  + \
               self.log_proposal_copula_l(t,x_curr,x_prev,observ,prop_copula_params)

    def log_normal(self, x, mu, Sigma):
        dim = Sigma.get_shape().as_list()[0]
        sign, logdet = tf.linalg.slogdet(Sigma)
        log_norm = -0.5*dim*np.log(2.*np.pi) - 0.5*logdet
        Prec = tf.dtypes.cast(tf.linalg.inv(Sigma), dtype=tf.float32)
        # print(mu)
        # print(mu.get_shape().as_list()[0])
        first_term = x - mu
        # print(first_term.get_shape().as_list())
        second_term = tf.transpose(tf.matmul(Prec, tf.transpose(x-mu)))
        ls_term = -0.5*tf.reduce_sum(first_term*second_term,1)
        # print(ls_term.get_shape().as_list())
        return tf.cast(log_norm, dtype=tf.float32) + tf.cast(ls_term, dtype=tf.float32)

    def log_target(self, t, x_curr, x_prev, observ):
        init_pose, init_cov, A, Q, C, R = self.target_params
        if t > 0:
            logF = self.log_normal(x_curr, tf.transpose(self.transition_model(tf.transpose(x_prev))), Q)
        else:
            logF = self.log_normal(x_curr, tf.transpose(init_pose), init_cov)
        logG = self.log_normal(tf.transpose(self.measurement_model(tf.transpose(x_curr))), tf.transpose(tf.convert_to_tensor(observ[t], dtype=tf.float32)), R)
        return logF + logG

    def log_proposal_marginal(self, t, x_curr, x_prev, observ, prop_marg_params):
        """
        In the Gaussian case the multivariate normal with diagonal covariance equals:
        prod_{i=1}^{self.state_dim} p(x_curr[i] | x_prev[i]; proposal_params)
        """
        init_pose, init_cov, A, Q, C, R = self.target_params
        mut = prop_marg_params[t,0:3]
        lint = prop_marg_params[t,3:6]
        log_s2t = prop_marg_params[t,6:9]
        s2t = tf.exp(log_s2t)
        if t > 0:
            mu = mut + tf.transpose(self.transition_model(tf.transpose(x_prev)))*lint
        else:
            mu = mut + lint*tf.transpose(init_pose)
        return self.log_normal(x_curr, mu, tf.diag(s2t))

    def log_proposal(self, t, x_curr, x_prev, observ, proposal_params):
        prop_copula_params, prop_marg_params = proposal_params
        return self.log_proposal_copula(t, x_curr, x_prev, observ, prop_copula_params) + \
               self.log_proposal_marginal(t, x_curr, x_prev, observ, prop_marg_params)

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
    num_steps = 2
    # Number of particles to use during training
    num_train_particles = 1000
    # Number of particles to use during SMC query
    num_query_particles = 1000000
    # Number of iterations to fit the proposal parameters
    num_train_steps = 1000
    # Learning rate for the distribution
    lr_m = 0.001
    # Number of random seeds for experimental trials
    num_seeds = 1
    # Number of samples to use for plotting
    num_samps = 10000
    # Proposal initial scale
    prop_scale = -0.01


    # True target parameters
    # Consider replacing this with "map", "initial_pose", "true_measurement_model", and "true_odometry_model"
    init_pose = tf.ones([3,1],dtype=np.float32)
    init_cov = 0.01*tf.eye(3,3,dtype=np.float32)
    A = 1.1*tf.eye(3,3,dtype=np.float32)
    Q = 0.01*tf.eye(3,3,dtype=np.float32)
    C = tf.eye(2,3,dtype=np.float32)
    R = 0.01*tf.eye(2,2,dtype=np.float32)
    target_params = [init_pose,init_cov,A,Q,C,R]

    # Create the session
    sess = tf.Session()

    # Create the agent
    rs = np.random.RandomState(1)# This remains fixed for the ground truth
    td_agent = RangeBearingAgent(target_params=target_params, rs=rs, num_steps=num_steps, prop_scale=prop_scale)

    # Generate observations TODO: change to numpy implementation
    x_true, z_true = td_agent.generate_data()
    xt_vals, zt_vals = sess.run([x_true, z_true])

    # Get posterior samples (since everything is linear Gaussian, just do Kalman filtering)
    # TODO: change to numpy implementation
    post_mean, post_cov = td_agent.lgss_posterior_params(zt_vals, 1)
    p_mu, p_cov = sess.run([post_mean, post_cov])
    post_values = td_agent.rs.multivariate_normal(mean=p_mu.ravel(), cov=p_cov, size=num_samps)
    post_values = np.array(post_values).reshape((num_samps, td_agent.state_dim))

    for seed in range(num_seeds):
        sess = tf.Session()
        tf.set_random_seed(seed)

        # Summary writer
        writer = tf.summary.FileWriter('./logs', sess.graph)

        # Create the VCSLAM instance with above parameters
        vcs = VCSLAM(sess = sess,
                     vcs_agent = td_agent,
                     observ = zt_vals,
                     num_particles = num_train_particles,
                     num_train_steps = num_train_steps,
                     lr_m = lr_m,
                     summary_writer = writer)

        # Train the model
        opt_proposal_params, train_sess = vcs.train(vcs_agent = td_agent)
        opt_proposal_params = train_sess.run(opt_proposal_params)
        print(opt_proposal_params)

        # Sample the model
        my_vars = vcs.sim_q(opt_proposal_params, target_params, zt_vals, td_agent, num_samples=num_samps, num_particles=num_query_particles)

        # temporary
        # my_vars = vcs.sim_q(opt_propsal_params, target_params, zt_vals, td_agent, num_samples=1, num_particles=num_query_particles)
        # my_samples = [train_sess.run(my_vars) for i in range(num_samps)]
        my_samples = train_sess.run(my_vars)
        samples_np = np.squeeze(np.array(my_samples))
        print(samples_np.shape)
        plotting.plot_dist(samples_np,post_values)

        # plots TODO: clean up more and add other relevant plots
        xt_vals = np.array(xt_vals).reshape(td_agent.num_steps, td_agent.state_dim)
        zt_vals = np.array(zt_vals)
        plotting.plot_kde(samples_np,post_values,xt_vals,zt_vals)
        plotting.plot_dist(samples_np,post_values)
        # plt.show()

