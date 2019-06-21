import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow.contrib.distributions as tfd
import plotting
import tensorflow_probability as tfp
# tfb = tfp.bijectors
import correlation_cholesky as cc

from vcsmc import *
import vcslam_agent

import copula_gaussian as cg

# Remove warnings
tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()

class ThreeDoorsAgent(vcslam_agent.VCSLAMAgent):
    def __init__(self,
                 target_params,
                 num_steps=3,
                 state_dim=1,
                 num_landmarks=2,
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
        self.copula_s = cg.WarpedGaussianCopula(
            loc = tf.zeros(shape=self.state_dim, dtype=tf.float32),
            scale_tril = tf.eye(self.state_dim, dtype=tf.float32),
            marginal_bijectors=[
                cg.NormalCDF(loc=0., scale=1.),
                cg.NormalCDF(loc=0., scale=1.),
                cg.NormalCDF(loc=0., scale=1.)])

    def transition_model(self, x):
        lm1_prior_mean, lm1_prior_var, lm2_prior_mean, lm2_prior_var, lm3_prior_mean, lm3_prior_var, motion_mean, motion_var, meas_var = self.target_params
        return x + motion_mean

    def measurement_model(self, x):
        init_pose, init_cov, A, Q, C, R = self.target_params
        return tf.matmul(C, x)

    def get_dependency_param_shape(self):
        return [self.num_steps, self.state_dim]

    def get_marginal_param_shape(self):
        return [self.num_steps, self.state_dim*3]

    def init_marg_params(self):
        T = self.num_steps
        Dx = self.state_dim
        marg_params = np.array([np.array([self.prop_scale * self.rs.randn(Dx), # Bias
                 1. + self.prop_scale * self.rs.randn(Dx)]).ravel() # Linear times A/mu0
                for t in range(T)])
        return marg_params

    def init_dependency_params(self):
        # State-component copula model represents a joint dependency distribution
        # over the state components
        T = self.num_steps
        Dx = self.state_dim
        # Each correlation param should be in [-1,1] So we compute the actual
        # correlation param as (1 - exp(-x)) / (1 + exp(-x)) which you can
        # verify as 2*sigmoid(x) - 1 where sigmoid(x): R -> [0,1] = 1 / (1 +
        # exp(-x)) and we just rescale that to be [-1,1] I think there's a
        # better way I just need to think about it for a bit - Kevin
        copula_params = np.array([np.array(self.cop_scale * self.rs.randn(Dx)).ravel() # correlation/covariance params
                                  for t in range(T)])
        # copula_params = np.array([np.zeros(Dx).ravel() for t in range(T)])
        return copula_params

    def generate_data(self):
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
        prop_copula_params = proposal_params[0]
        prop_marg_params = proposal_params[1]

        mut = prop_marg_params[t,0:3]
        lint = prop_marg_params[t,3:6]

        # Use "bootstrap" method for s2t of univariates
        s2t = tf.diag_part(Q)

        if t > 0:
            mu = mut + tf.transpose(self.transition_model(tf.transpose(x_prev)))*lint
        else:
            mu = mut + lint*tf.transpose(init_pose)

        # Copula params are defined over the reals, but we want a correlation matrix
        # So we use a sigmoid map to take reals to the range [-1,1]
        # r_vec = (1. - tf.exp(-prop_copula_params[t,:]))/(1. + tf.exp(-prop_copula_params[t,:])) # should be length
        r_vec = prop_copula_params[t,:]
        L_mat = cc.CorrelationCholesky().forward(r_vec)

        # Marginal bijectors will be the CDFs of the univariate marginals Here
        # these are normal CDFs
        # x1_cdf = cg.NormalCDF(loc=tf.transpose([mu[:,0]]), scale=tf.sqrt(s2t[0]))
        # x2_cdf = cg.NormalCDF(loc=tf.transpose([mu[:,1]]), scale=tf.sqrt(s2t[1]))
        # x3_cdf = cg.NormalCDF(loc=tf.transpose([mu[:,2]]), scale=tf.sqrt(s2t[2]))
        # x_cdf = cg.MixtureCDF(loc=tf.transpose([mu[]]))
        x_cdf = cg.GaussianMixtureCDF(ps=[1.], locs=[mu1, mu2, mu3], scales=tf.sqrt(s2t[0]))
        l1_cdf = cg.NormalCDF(loc=mul1, scale=tf.sqrt(s2t[0]))
        l2_cdf = cg.NormalCDF(loc=mul2, scale=tf.sqrt(s2t[0]))
        l3_cdf = cg.NormalCDF(loc=mul3, scale=tf.sqrt(s2t[0]))

        # Build a copula (can also store globally if we want) we would just
        #  have to modify self.copula.scale_tril and
        #  self.copula.marginal_bijectors in each iteration NOTE: I add
        #  tf.eye(3) to L_mat because I think the diagonal has to be > 0
        gc = cg.WarpedGaussianCopula(
            loc=[0., 0., 0.],
            scale_tril=L_mat, # TODO This is currently just eye(3), use L_mat!
            marginal_bijectors=[
                x_cdf,
                l1_cdf,
                l2_cdf,
                l3_cdf])
        # self.copula_s._bijector = cg.Concat([x1_cdf, x2_cdf, x3_cdf])
        # self.copula_s.distribution.scale_tril = L_mat

        # sample = self.copula_s.sample(x_prev.get_shape().as_list()[0])
        return gc.sample(x_prev.get_shape().as_list()[0])

    def log_proposal_copula_sl(self,t,x_curr,x_prev,observ,proposal_params):
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

    def log_proposal_copula_ll(self,t,x_curr,x_prev,observ,proposal_params):
        """
        Log probability from the landmark-landmark copula
        """
        num_particles = x_curr.get_shape().as_list()[0]
        return tf.zeros(shape=(num_particles),dtype=tf.float32)

    def log_proposal_copula_s(self,t,x_curr,x_prev,observ,proposal_params):
        """
        Log probability from the state-component copula
        """
        # TODO: implement Gaussian copula here

        init_pose, init_cov, A, Q, C, R = self.target_params
        prop_copula_params, prop_marg_params = proposal_params

        mut = prop_marg_params[t,0:3]
        lint = prop_marg_params[t,3:6]
        s2t = tf.diag_part(Q)

        # Copula params are defined over the reals, but we want a correlation matrix
        # So we use a sigmoid map to take reals to the range [-1,1]
        # r_vec = (1. - tf.exp(-prop_copula_params[t,:]))/(1. + tf.exp(-prop_copula_params[t,:])) # should be length
        # use correlationcholesky to do the right thing
        r_vec = prop_copula_params[t,:]
        L_mat = cc.CorrelationCholesky().forward(r_vec)

        # This is how we properly compute the marginal parameters based on the
        # model c.f. log_proposal_marginal function
        if t > 0:
            mu = mut + tf.transpose(self.transition_model(tf.transpose(x_prev)))*lint
        else:
            mu = mut + lint*tf.transpose(init_pose)

        # Marginal bijectors will be the CDFs of the univariate marginals Here
        # these are normal CDFs and GaussianMixtureCDF
        x_cdf = cg.GaussianMixtureCDF(ps=[1.], locs=[mu1, mu2, mu3], scales=tf.sqrt(s2t[0]))
        l1_cdf = cg.NormalCDF(loc=mul1, scale=tf.sqrt(s2t[0]))
        l2_cdf = cg.NormalCDF(loc=mul2, scale=tf.sqrt(s2t[0]))
        l3_cdf = cg.NormalCDF(loc=mul3, scale=tf.sqrt(s2t[0]))

        # Build a copula (can also store globally if we want) we would just
        #  have to modify self.copula.scale_tril and
        #  self.copula.marginal_bijectors in each iteration NOTE: I add
        #  tf.eye(3) to L_mat because I think the diagonal has to be > 0
        gc = cg.WarpedGaussianCopula(
            loc=[0., 0., 0.],
            scale_tril=L_mat, # TODO This is currently just eye(3), use L_mat!
            marginal_bijectors=[
                x_cdf,
                l1_cdf,
                l2_cdf,
                l3_cdf])

        return gc.log_prob(x_curr)


    def log_proposal_copula_l(self,t,x_curr,x_prev,observ,proposal_params):
        """
        Log probability from the landmark-component copula
        """
        num_particles = x_curr.get_shape().as_list()[0]
        return tf.zeros(shape=(num_particles),dtype=tf.float32)

    def log_proposal_copula(self,t,x_curr,x_prev,observ,proposal_params):
        """
        Returns the log probability from the copula model described in Equation 4
        """
        return self.log_proposal_copula_sl(t,x_curr,x_prev,observ,proposal_params) + \
            self.log_proposal_copula_ll(t,x_curr,x_prev,observ,proposal_params) + \
            self.log_proposal_copula_s(t,x_curr,x_prev,observ,proposal_params) + \
            self.log_proposal_copula_l(t,x_curr,x_prev,observ,proposal_params)

    def log_normal(self, x, mu, Sigma):
        dim = Sigma.get_shape().as_list()[0]
        sign, logdet = tf.linalg.slogdet(Sigma)
        log_norm = -0.5*dim*np.log(2.*np.pi) - 0.5*logdet
        Prec = tf.dtypes.cast(tf.linalg.inv(Sigma), dtype=tf.float32)
        first_term = x - mu
        second_term = tf.transpose(tf.matmul(Prec, tf.transpose(x-mu)))
        ls_term = -0.5*tf.reduce_sum(first_term*second_term,1)
        return tf.cast(log_norm, dtype=tf.float32) + tf.cast(ls_term, dtype=tf.float32)

    def log_target(self, t, x_curr, x_prev, observ):
        # init_pose, init_cov, A, Q, C, R = self.target_params
        lm1_prior_mean, lm1_prior_var, lm2_prior_mean, lm2_prior_var, lm3_prior_mean, lm3_prior_var, motion_mean, motion_var, meas_var = self.target_params
        if t > 0:
            logF = self.log_normal(x_curr, tf.transpose(self.transition_model(tf.transpose(x_prev))), motion_var)
        if t == 0 or t == 1:
            logG = tf.log((1./3.)*tf.exp(self.log_normal(x_curr[:,0], x_curr[:,1], meas_var)) + \
                          (1./3.)*tf.exp(self.log_normal(x_curr[:,0], x_curr[:,2], meas_var)) + \
                          (1./3.)*tf.exp(self.log_normal(x_curr[:,0], x_curr[:,3], meas_var)))
        logH = self.log_normal(x_curr[:,1], lm1_prior_mean, lm1_prior_var) + \
               self.log_normal(x_curr[:,2], lm2_prior_mean, lm2_prior_var) + \
               self.log_normal(x_curr[:,3], lm3_prior_mean, lm3_prior_var)
        return logF + logG + logH

    def log_proposal(self, t, x_curr, x_prev, observ, proposal_params):
        """
            Note: we don't actually need log_proposal_marginal
            This is because log_proposal_copula computes the whole thing
        """
        prop_copula_params, prop_marg_params = proposal_params
        prop_copula_params = tf.debugging.check_numerics(prop_copula_params, "Copula param error")
        prop_marg_params = tf.debugging.check_numerics(prop_marg_params, "Marg param error")
        cl = self.log_proposal_copula(t, x_curr, x_prev, observ, proposal_params)
        return cl

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
    num_steps = 10
    # Number of particles to use during training
    num_train_particles = 100
    # Number of particles to use during SMC query
    num_query_particles = 1000000
    # Number of iterations to fit the proposal parameters
    num_train_steps = 2000
    # Learning rate for the marginal
    lr_m = 0.01
    # Learning rate for the copula
    lr_d = 0.001
    # Number of random seeds for experimental trials
    num_seeds = 1
    # Number of samples to use for plotting
    num_samps = 10000
    # Proposal initial scale
    prop_scale = 0.5
    # Copula initial scale
    cop_scale = 0.1


    # True target parameters
    # Consider replacing this with "map", "initial_pose", "true_measurement_model", and "true_odometry_model"

    lm1_prior_mean, lm1_prior_var, lm2_prior_mean, lm2_prior_var, lm3_prior_mean, lm3_prior_var, motion_var, meas_var = self.target_params
    lm1_prior_mean = 0.0
    lm1_prior_var = 0.01
    lm2_prior_mean = 2.0
    lm2_prior_var = 0.01
    lm3_prior_mean = 4.0
    lm3_prior_var = 0.01
    A = 1.1*tf.eye(3,3,dtype=np.float32)
    target_params = [lm1_prior_mean, lm1_prior_var,
                     lm2_prior_mean, lm2_prior_var,
                     lm3_prior_mean, lm3_prior_var,
                     motion_mean, motion_var,
                     meas_var]

    # Create the session
    sess = tf.Session()

    # Create the agent
    rs = np.random.RandomState(1)# This remains fixed for the ground truth
    td_agent = RangeBearingAgent(target_params=target_params, rs=rs, num_steps=num_steps, prop_scale=prop_scale, cop_scale=cop_scale)

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
                     lr_d = lr_d,
                     lr_m = lr_m,
                     summary_writer = writer)

        # Train the model
        opt_proposal_params, train_sess = vcs.train(vcs_agent = td_agent)
        opt_proposal_params = train_sess.run(opt_proposal_params)
        opt_dep_params, opt_marg_params = opt_proposal_params
        print(opt_proposal_params)
        print("Optimal dep params: ", opt_dep_params)

        # Sample the model
        my_vars = vcs.sim_q(opt_proposal_params, target_params, zt_vals, td_agent, num_samples=num_samps, num_particles=num_query_particles)

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

