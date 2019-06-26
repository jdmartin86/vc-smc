import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow.contrib.distributions as tfd
import plotting
import tensorflow_probability as tfp
# tfb = tfp.bijectors
import correlation_cholesky as cc
from scipy.special import comb

from vcsmc import *
import vcslam_agent

# Temporary
import seaborn as sbs
import matplotlib.pyplot as plt

import copula_gaussian as cg

# Remove warnings
tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()

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
        # if not latent_dim:
        #     self.latent_dim = self.num_steps * (self.state_dim + self.landmark_dim*self.num_landmarks)
        # else:
        #     self.latent_dim = latent_dim
        self.latent_dim = (self.state_dim + self.landmark_dim*self.num_landmarks)
        # A copula over N variables has (N choose 2) correlation parameters
        self.copula_dim = int(comb(self.latent_dim,2))
        # Observation dimensionality (direct observations of x_door)
        self.observ_dim = observ_dim
        # Proposal params
        self.proposal_params = tf.placeholder(dtype=tf.float32,shape=(10,1))

        self.prop_scale = prop_scale
        self.cop_scale = cop_scale

        # Random state for sampling
        self.rs = rs

        # initialize dependency models
        # self.copula_s = cg.WarpedGaussianCopula(
        #     loc = tf.zeros(shape=self.state_dim, dtype=tf.float32),
        #     scale_tril = tf.eye(self.state_dim, dtype=tf.float32),
        #     marginal_bijectors=[
        #         cg.NormalCDF(loc=0., scale=1.),
        #         cg.NormalCDF(loc=0., scale=1.),
        #         cg.NormalCDF(loc=0., scale=1.)])

    def transition_model(self, x):
        init_mean, init_var, lm1_prior_mean, lm1_prior_var, lm2_prior_mean, lm2_prior_var, lm3_prior_mean, lm3_prior_var, motion_mean, motion_var, meas_var = self.target_params
        return x + motion_mean

    def measurement_model(self, x):
        return 0.

    def get_dependency_param_shape(self):
        return [self.num_steps, self.copula_dim]

    def get_marginal_param_shape(self):
        return [self.num_steps+1, self.state_dim*3]

    def init_marg_params(self):
        T = self.num_steps
        Dx = self.state_dim
        """
        marginal params for this problem are the means of the Gaussians at each time step (so 3 gaussians * T time steps) as well as the landmark location means (so num landmarks)

        might also add the bias term?

        """
        # marg_params = np.array([np.array([self.prop_scale * self.rs.randn(Dx), # Bias
        #          1. + self.prop_scale * self.rs.randn(Dx)]).ravel() # Linear times A/mu0
        #         for t in range(T)])
        # marg_params = np.array([np.array([self.prop_scale * self.rs.randn(Dx), # Bias
        #                                   1. + self.prop_scale * self.rs.randn(Dx)]).ravel() # Linear times A/mu0
        #                         for t in range(T)]
        #                        .extend([self.rs.randn(Dl)]))
        # marg_params = np.array([np.array([self.prop_scale * self.rs.randn(4*Dx)]).ravel() # 3 MoG means per time step
        #                                   for t in range(T+1)])

        marg_params = np.array([np.array([self.prop_scale * self.rs.randn(3*Dx)]).ravel() # 3 MoG means per time step
                                          for t in range(T+1)])

        # marg_params = np.array([np.array([0.0, 2.0, 6.0]).ravel() # 3 MoG means per time step
        #                                   for t in range(T+1)])
        return marg_params

    def init_dependency_params(self):
        # Copula model represents a joint dependency distribution
        # over the latent variable components
        T = self.num_steps
        copula_params = np.array([np.array(self.cop_scale * self.rs.randn(self.copula_dim)).ravel() # correlation params
                                  for t in range(T)])
        return copula_params

    # def generate_data(self):
    #     init_pose, init_cov, A, Q, C, R = self.target_params
    #     Dx = init_pose.get_shape().as_list()[0]
    #     Dz = R.get_shape().as_list()[0]

    #     x_true = []
    #     z_true = []

    #     for t in range(self.num_steps):
    #         if t > 0:
    #             x_true.append(tf.transpose(tfd.MultivariateNormalFullCovariance(loc=tf.transpose(self.transition_model(x_true[t-1])),
    #                                                                covariance_matrix=Q).sample(seed=self.rs.randint(0,1234))))
    #         else:
    #             x_sample = init_pose
    #             x_true.append(x_sample)
    #         z_true.append(tf.transpose(tfd.MultivariateNormalFullCovariance(loc=tf.transpose(self.measurement_model(x_true[t])),
    #                                                            covariance_matrix=R).sample(seed=self.rs.randint(0,1234))))

    #     return x_true, z_true

    def sim_proposal(self, t, x_prev, observ, proposal_params):
        init_mean, init_var, lm1_prior_mean, lm1_prior_var, lm2_prior_mean, lm2_prior_var, lm3_prior_mean, lm3_prior_var, motion_mean, motion_var, meas_var = self.target_params
        T = self.num_steps
        num_particles = x_prev.get_shape().as_list()[0]
        prop_copula_params = proposal_params[0]
        prop_marg_params = proposal_params[1]

        mu1t = prop_marg_params[t,0]
        mu2t = prop_marg_params[t,1]
        mu3t = prop_marg_params[t,2]
        # mu4t = prop_marg_params[t,3]
        # logs2t = prop_marg_params[t,3]
        l1m = prop_marg_params[T,0]
        l2m = prop_marg_params[T,1]
        l3m = prop_marg_params[T,2]

        # l1m = lm1_prior_mean
        # l2m = lm2_prior_mean
        # l3m = lm3_prior_mean

        if t == 0:
            mu1 = mu1t
            mu2 = mu2t
            mu3 = mu3t
            # mu4 = mu4t
            # mu1 = lm1_prior_mean[0,0]
            # mu2 = lm2_prior_mean[0,0]
            # mu3 = lm3_prior_mean[0,0]
        if t > 0:
            transition = tf.transpose(self.transition_model(tf.transpose(x_prev[:,0])))
            mu1 = mu1t + transition
            mu2 = mu2t + transition
            mu3 = mu3t + transition

        # if t > 0:
        #     mu = mut + tf.transpose(self.transition_model(tf.transpose(x_prev)))*lint
        # else:
        #     mu = mut + lint*tf.transpose(init_pose)

        # Copula params are defined over the reals, but we want a correlation matrix
        # So we use a sigmoid map to take reals to the range [-1,1]
        # r_vec = (1. - tf.exp(-prop_copula_params[t,:]))/(1. + tf.exp(-prop_copula_params[t,:])) # should be length
        r_vec = prop_copula_params[t,:]
        L_mat = cc.CorrelationCholesky().forward(r_vec)
        print("L Mat shape: ", L_mat.get_shape().as_list())

        # Marginal bijectors will be the CDFs of the univariate marginals Here
        # these are normal CDFs
        # x_cdf = cg.GaussianMixtureCDF(ps=[1./3., 1./3., 1./3.], locs=[mu1, mu2, mu3], scales=[tf.sqrt(motion_var), tf.sqrt(motion_var), tf.sqrt(motion_var)])
        # x_cdf = cg.GaussianMixtureCDF(ps=[1.], locs=[mu1], scales=[tf.sqrt(motion_var)])
        # x_cdf = cg.EmpGaussianMixtureCDF()
        # x_scale = tf.sqrt(motion_var)
        # x_scale = tf.sqrt(tf.exp(logs2t))
        if t == 0:
            # x_scale = tf.sqrt(4.0*meas_var)
            x_scale = tf.sqrt(init_var)
        if t > 0:
            x_scale = tf.sqrt(motion_var)
        # print("mu1 shape: ", mu1.get_shape().as_list())
        # print("mu2 shape: ", mu2.get_shape().as_list())
        # print("mu3 shape: ", mu3.get_shape().as_list())
        # ps = tf.transpose(tf.constant([[1./3.], [1./3.], [1./3.]]))
        if t == 0:
            ps = [1./3., 1./3., 1./3.]
        if t > 0:
            ps = tf.constant([1./3., 1./3., 1./3.])*tf.ones_like(mu1)
            mu1 = mu1[:,0]
            mu2 = mu2[:,0]
            mu3 = mu3[:,0]
        print("P vector: ", t, ps)
        # print("mu1 vector", mu1[:,0])
        # ps = [0.25, 0.25, 0.25, 0.25]
        # ps = [1.0]
        x_cdf = cg.EmpGaussianMixtureCDF(ps=ps,
                                         locs=[mu1, mu2, mu3],
                                         scales=[x_scale[0,0], x_scale[0,0], x_scale[0,0]])

        # x_cdf = cg.EmpGaussianMixtureCDF(ps=ps,
        #                                  locs=[mu1, mu2, mu3, mu4],
        #                                  scales=[x_scale[0,0], x_scale[0,0], x_scale[0,0], x_scale[0,0]])


        # x_cdf = cg.NormalCDF(loc=mu1, scale=x_scale)
        l1_cdf = cg.NormalCDF(loc=l1m, scale=tf.sqrt(lm1_prior_var))
        l2_cdf = cg.NormalCDF(loc=l2m, scale=tf.sqrt(lm2_prior_var))
        l3_cdf = cg.NormalCDF(loc=l3m, scale=tf.sqrt(lm3_prior_var))

        # Build a copula (can also store globally if we want) we would just
        #  have to modify self.copula.scale_tril and
        #  self.copula.marginal_bijectors in each iteration NOTE: I add
        #  tf.eye(3) to L_mat because I think the diagonal has to be > 0
        gc = cg.WarpedGaussianCopula(
            loc=[0., 0., 0., 0.],
            scale_tril=L_mat,
            marginal_bijectors=[
                x_cdf,
                l1_cdf,
                l2_cdf,
                l3_cdf])
        # self.copula_s._bijector = cg.Concat([x1_cdf, x2_cdf, x3_cdf])
        # self.copula_s.distribution.scale_tril = L_mat

        # sample = self.copula_s.sample(x_prev.get_shape().as_list()[0])
        return gc.sample(x_prev.get_shape().as_list()[0])

    def log_proposal(self,t,x_curr,x_prev,observ,proposal_params):
        """
        Log probability from the state-component copula
        """
        prop_copula_params, prop_marg_params = proposal_params

        # Extract params here
        init_mean, init_var, lm1_prior_mean, lm1_prior_var, lm2_prior_mean, lm2_prior_var, lm3_prior_mean, lm3_prior_var, motion_mean, motion_var, meas_var = self.target_params

        T = self.num_steps
        num_particles = x_prev.get_shape().as_list()[0]
        prop_copula_params = proposal_params[0]
        prop_marg_params = proposal_params[1]

        mu1t = prop_marg_params[t,0]
        mu2t = prop_marg_params[t,1]
        mu3t = prop_marg_params[t,2]
        # mu4t = prop_marg_params[t,3]
        # logs2t = prop_marg_params[t,3]
        l1m = prop_marg_params[T,0]
        l2m = prop_marg_params[T,1]
        l3m = prop_marg_params[T,2]

        if t == 0:
            mu1 = mu1t
            mu2 = mu2t
            mu3 = mu3t
            # mu4 = mu4t
            # mu1 = lm1_prior_mean[0,0]
            # mu2 = lm2_prior_mean[0,0]
            # mu3 = lm3_prior_mean[0,0]
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
        print("P vector: ", t, ps)
        # print("mu1 vector", mu1[:,0])
        # ps = [0.25, 0.25, 0.25, 0.25]
        # ps = [1.0]
        x_cdf = cg.EmpGaussianMixtureCDF(ps=ps,
                                         locs=[mu1, mu2, mu3],
                                         scales=[x_scale[0,0], x_scale[0,0], x_scale[0,0]])

        # x_cdf = cg.EmpGaussianMixtureCDF(ps=ps,
        #                                  locs=[mu1, mu2, mu3, mu4],
        #                                  scales=[x_scale[0,0], x_scale[0,0], x_scale[0,0], x_scale[0,0]])

        # x_cdf = cg.EmpGaussianMixtureCDF(ps=ps,
        #                                  locs=[mu1,mu2,mu3],
        #                                  scales=[x_scale, x_scale, x_scale])

        # x_cdf = cg.NormalCDF(loc=mu1, scale=x_scale)
        l1_cdf = cg.NormalCDF(loc=l1m, scale=tf.sqrt(lm1_prior_var))
        l2_cdf = cg.NormalCDF(loc=l2m, scale=tf.sqrt(lm2_prior_var))
        l3_cdf = cg.NormalCDF(loc=l3m, scale=tf.sqrt(lm3_prior_var))

        # x_cdf = cg.NormalCDF(loc=mu1, scale=tf.sqrt(motion_var))
        # l1_cdf = cg.NormalCDF(loc=l1m, scale=tf.sqrt(lm1_prior_var))
        # l2_cdf = cg.NormalCDF(loc=l2m, scale=tf.sqrt(lm2_prior_var))
        # l3_cdf = cg.NormalCDF(loc=l3m, scale=tf.sqrt(lm3_prior_var))

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
        dim = Sigma.get_shape().as_list()[0]
        sign, logdet = tf.linalg.slogdet(Sigma)
        log_norm = -0.5*dim*np.log(2.*np.pi) - 0.5*logdet
        Prec = tf.dtypes.cast(tf.linalg.inv(Sigma), dtype=tf.float32)
        first_term = x - mu
        second_term = tf.transpose(tf.matmul(Prec, tf.transpose(x-mu)))
        ls_term = -0.5*tf.reduce_sum(first_term*second_term,1)
        return tf.cast(log_norm, dtype=tf.float32) + tf.cast(ls_term, dtype=tf.float32)

    def log_target(self, t, x_curr, x_prev, observ):
        init_mean, init_var, lm1_prior_mean, lm1_prior_var, lm2_prior_mean, lm2_prior_var, lm3_prior_mean, lm3_prior_var, motion_mean, motion_var, meas_var = self.target_params
        print("X curr shape: ", x_curr.get_shape().as_list())
        x_prev_samples = tf.transpose(tf.gather_nd(tf.transpose(x_prev),[[0]]))
        x_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[0]]))
        l1_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[1]]))
        l2_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[2]]))
        l3_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[3]]))
        print("x sample shape: ", x_samples.get_shape().as_list())
        if t > 0:
            logG = self.log_normal(x_samples, tf.transpose(self.transition_model(tf.transpose(x_prev_samples))), motion_var)
        if t == 0 or t == 1:
            logG = tf.log((1./3.)*tf.exp(self.log_normal(x_samples, l1_samples, meas_var)) + \
                          (1./3.)*tf.exp(self.log_normal(x_samples, l2_samples, meas_var)) + \
                          (1./3.)*tf.exp(self.log_normal(x_samples, l3_samples, meas_var)))
        logH = self.log_normal(l1_samples, lm1_prior_mean, lm1_prior_var) + \
               self.log_normal(l2_samples, lm2_prior_mean, lm2_prior_var) + \
               self.log_normal(l3_samples, lm3_prior_mean, lm3_prior_var)
        return logG + logH

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
    num_train_steps = 2000
    # Learning rate for the marginal
    lr_m = 0.05
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
    lm1_prior_mean = tf.zeros([1,1],dtype=tf.float32)
    lm1_prior_var = 0.01*tf.ones([1,1],dtype=tf.float32)
    lm2_prior_mean = 2.0*tf.ones([1,1],dtype=tf.float32)
    lm2_prior_var = 0.01*tf.ones([1,1],dtype=tf.float32)
    lm3_prior_mean = 5.0*tf.ones([1,1],dtype=tf.float32)
    lm3_prior_var = 0.01*tf.ones([1,1],dtype=tf.float32)
    motion_mean = 2.0*tf.ones([1,1],dtype=tf.float32)
    motion_var = 0.1*tf.ones([1,1],dtype=tf.float32)
    meas_var = 0.1*tf.ones([1,1],dtype=tf.float32)
    init_mean = tf.zeros([1,1],dtype=tf.float32)
    init_var = 5.0*meas_var
    target_params = [init_mean, init_var,
                     lm1_prior_mean, lm1_prior_var,
                     lm2_prior_mean, lm2_prior_var,
                     lm3_prior_mean, lm3_prior_var,
                     motion_mean, motion_var,
                     meas_var]

    # Create the session
    sess = tf.Session()

    # Create the agent
    rs = np.random.RandomState(1)# This remains fixed for the ground truth
    td_agent = ThreeDoorsAgent(target_params=target_params, rs=rs, num_steps=num_steps, prop_scale=prop_scale, cop_scale=cop_scale)

    # Generate observations TODO: change to numpy implementation
    # x_true, z_true = td_agent.generate_data()
    # xt_vals, zt_vals = sess.run([x_true, z_true])
    zt_vals = None

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

        # plots TODO: clean up more and add other relevant plots
        # xt_vals = np.array(xt_vals).reshape(td_agent.num_steps, td_agent.state_dim)
        # zt_vals = np.array(zt_vals)
        # plotting.plot_kde(samples_np,None,xt_vals,zt_vals)
        # plotting.plot_dist(samples_np,None)
        sbs.distplot(samples_np[:,0], color='purple')
        sbs.distplot(samples_np[:,1], color='red')
        sbs.distplot(samples_np[:,2], color='green')
        sbs.distplot(samples_np[:,3], color='blue')

        plt.show()

