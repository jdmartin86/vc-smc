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

from bpf import *
import vcslam_agent

# Temporary
import seaborn as sbs
import matplotlib.pyplot as plt

import copula_gaussian as cg

# For evaluation
import metrics

# Remove warnings
tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()

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
        return [self.num_steps+1, self.state_dim*4]

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
        # initial_x = self.prop_scale*self.rs.randn(1)
        marg_params = np.array([np.array([0.0, 0.0, 2.0, 6.0]).ravel() # 3 MoG means per time step
                                          for t in range(T+1)])
        return marg_params

    def init_dependency_params(self):
        # Copula model represents a joint dependency distribution
        # over the latent variable components
        T = self.num_steps
        copula_params = np.array([np.array(self.cop_scale * self.rs.randn(self.copula_dim)).ravel() # correlation params
                                  for t in range(T)])
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
            #z_true.append(tf.transpose(tfd.MultivariateNormalFullCovariance(loc=tf.transpose(self.measurement_model(x_true[t])),
                                                               #covariance_matrix=R).sample(seed=self.rs.randint(0,1234))))

        return x_true#, z_true

    def sim_proposal(self, t, x_prev, observ, proposal_params):
        init_mean, init_var, lm1_prior_mean, lm1_prior_var, lm2_prior_mean, lm2_prior_var, lm3_prior_mean, lm3_prior_var, motion_mean, motion_var, meas_var = self.target_params
        T = self.num_steps
        num_particles = x_prev.get_shape().as_list()[0]
        prop_copula_params = proposal_params[0]
        prop_marg_params = proposal_params[1]

        mu1t = prop_marg_params[t,0]
        l1m = prop_marg_params[T,1]
        l2m = prop_marg_params[T,2]
        l3m = prop_marg_params[T,3]

        if t == 0:
            mu1 = mu1t
        if t > 0:
            mu1 = mu1t + tf.transpose(self.transition_model(tf.transpose(x_prev[:,0])))

        if t == 0:
            x_scale = tf.sqrt(init_var)
        if t > 0:
            x_scale = tf.sqrt(motion_var)

        x_cdf = cg.NormalCDF(loc=mu1, scale=x_scale)
        l1_cdf = cg.NormalCDF(loc=l1m, scale=tf.sqrt(lm1_prior_var))
        l2_cdf = cg.NormalCDF(loc=l2m, scale=tf.sqrt(lm2_prior_var))
        l3_cdf = cg.NormalCDF(loc=l3m, scale=tf.sqrt(lm3_prior_var))

        # Build a copula (can also store globally if we want) we would just
        #  have to modify self.copula.scale_tril and
        #  self.copula.marginal_bijectors in each iteration NOTE: I add
        #  tf.eye(3) to L_mat because I think the diagonal has to be > 0
        gc = cg.WarpedGaussianCopula(
            loc=[0., 0., 0., 0.],
            scale_tril=tf.eye(4),
            marginal_bijectors=[
                x_cdf,
                l1_cdf,
                l2_cdf,
                l3_cdf])

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
        l1m = prop_marg_params[T,1]
        l2m = prop_marg_params[T,2]
        l3m = prop_marg_params[T,3]

        if t == 0:
            mu1 = mu1t
        if t > 0:
            transition = tf.transpose(self.transition_model(tf.transpose(x_prev[:,0])))
            mu1 = mu1t + transition

        if t == 0:
            x_scale = tf.sqrt(init_var)
        if t > 0:
            x_scale = tf.sqrt(motion_var)

        # Marginal bijectors will be the CDFs of the univariate marginals Here
        # these are normal CDFs and GaussianMixtureCDF
        # x_cdf = cg.GaussianMixtureCDF(ps=[1.], locs=[mu1, mu2, mu3], scales=[tf.sqrt(motion_var), tf.sqrt(motion_var), tf.sqrt(motion_var)])
        x_cdf = cg.NormalCDF(loc=mu1, scale=x_scale)
        l1_cdf = cg.NormalCDF(loc=l1m, scale=tf.sqrt(lm1_prior_var))
        l2_cdf = cg.NormalCDF(loc=l2m, scale=tf.sqrt(lm2_prior_var))
        l3_cdf = cg.NormalCDF(loc=l3m, scale=tf.sqrt(lm3_prior_var))

        # Build a copula (can also store globally if we want) we would just
        #  have to modify self.copula.scale_tril and
        #  self.copula.marginal_bijectors in each iteration NOTE: I add
        #  tf.eye(3) to L_mat because I think the diagonal has to be > 0
        gc = cg.WarpedGaussianCopula(
            loc=[0., 0., 0., 0.],
            scale_tril=tf.eye(4), # TODO This is currently just eye(3), use L_mat!
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
        # init_pose, init_cov, A, Q, C, R = self.target_params
        init_mean, init_var, lm1_prior_mean, lm1_prior_var, lm2_prior_mean, lm2_prior_var, lm3_prior_mean, lm3_prior_var, motion_mean, motion_var, meas_var = self.target_params
        print("X curr shape: ", x_curr.get_shape().as_list())
        x_prev_samples = tf.transpose(tf.gather_nd(tf.transpose(x_prev),[[0]]))
        x_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[0]]))
        l1_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[1]]))
        l2_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[2]]))
        l3_samples = tf.transpose(tf.gather_nd(tf.transpose(x_curr),[[3]]))
        print("x sample shape: ", x_samples.get_shape().as_list())
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
    #local_device_protos = device_lib.list_local_devices()
    #print([x.name for x in local_device_protos])
    # Optionally use accelerated computation
    # with tf.device("/device:XLA_CPU:0"):

    # Number of steps for the trajectory
    num_steps = 1
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
    num_seeds = 100
    # Number of samples to use for plotting
    num_samps = 2000
    # Proposal initial scale
    prop_scale = 2.0
    # Copula initial scale
    cop_scale = 0.1


    # True target parameters
    lm1_prior_mean = tf.zeros([1,1],dtype=tf.float32)
    lm1_prior_var = 0.01*tf.ones([1,1],dtype=tf.float32)
    lm2_prior_mean = 2.0*tf.ones([1,1],dtype=tf.float32)
    lm2_prior_var = 0.01*tf.ones([1,1],dtype=tf.float32)
    lm3_prior_mean = 6.0*tf.ones([1,1],dtype=tf.float32)
    lm3_prior_var = 0.01*tf.ones([1,1],dtype=tf.float32)
    motion_mean = 2.0*tf.ones([1,1],dtype=tf.float32)
    motion_var = 0.1*tf.ones([1,1],dtype=tf.float32)
    meas_var = 0.1*tf.ones([1,1],dtype=tf.float32)
    init_mean = tf.zeros([1,1], dtype=tf.float32)
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
    td_agent = ThreeDoorsGaussianBPFAgent(target_params=target_params, rs=rs, num_steps=num_steps, prop_scale=prop_scale, cop_scale=cop_scale)

    # Generate observations TODO: change to numpy implementation
    #x_true = td_agent.generate_data()
    #xt_vals = sess.run([x_true])
    truth = np.array([[0, 0, 2, 4],[2, 0, 2, 6],[4, 0, 2, 6]], np.int64)
    np.savetxt('output/trajectory_ref.txt', truth, delimiter=',')
    
    zt_vals = None

    all_kls = []

    error_1 = []
    mean_1 = []
    std_1 = []
    confidence_1 = []

    error_2 = []
    mean_2 = []
    std_2 = []
    confidence_2 = []

    error_3 = []
    mean_3 = []
    std_3 = []
    confidence_3 = []
    for seed in range(num_seeds):
        sess = tf.Session()
        tf.set_random_seed(seed)

        # Summary writer
        writer = tf.summary.FileWriter('./logs', sess.graph)

        # Create the VCSLAM instance with above parameters
        vcs = BootstrapParticleFilter(sess = sess,
                     vcs_agent = td_agent,
                     observ = zt_vals,
                     num_particles = num_train_particles,
                     num_train_steps = num_train_steps,
                     lr_d = lr_d,
                     lr_m = lr_m,
                     summary_writer = writer)

        # Train the model
        opt_proposal_params = vcs.train(vcs_agent = td_agent)
        opt_proposal_params = sess.run(opt_proposal_params)
        opt_dep_params, opt_marg_params = opt_proposal_params
        #print(opt_proposal_params)
        #print("Optimal dep params: ", opt_dep_params)

        # Sample the model
        my_vars = vcs.sim_q(opt_proposal_params, target_params, zt_vals, td_agent, num_samples=num_samps, num_particles=num_query_particles)

        my_samples = sess.run(my_vars)
        samples_np = np.squeeze(np.array(my_samples))
        #print(samples_np.shape)

        trajectory_mean = np.mean(samples_np, 0)
        trajectory_std = np.std(samples_np, 0)
        sq_errors = []
        conf_interval = []
        for i in range(4):
            #compute the mean squared error
            er_sq = (truth[num_steps-1,i]-trajectory_mean[i])**2
            sq_errors = np.append(sq_errors, er_sq)

            #compute the lower and upper bounds at 95% confidence interval
            lower_bound = trajectory_mean[i]-abs(1.96*(trajectory_std[i]/math.sqrt(num_samps)))
            upper_bound = trajectory_mean[i]+abs(1.96*(trajectory_std[i]/math.sqrt(num_samps)))
            conf_interval = np.append(conf_interval, [lower_bound, upper_bound])

        #save data based on how many steps
        if num_steps == 1:
            error_1 = np.r_[error_1, sq_errors]
            mean_1 = np.r_[mean_1, trajectory_mean]
            std_1 = np.r_[std_1, trajectory_std]
            confidence_1 = np.r_[confidence_1, conf_interval]
        elif num_steps == 2:
            error_2 = np.r_[error_2, sq_errors]
            mean_2 = np.r_[mean_2, trajectory_mean]
            std_2 = np.r_[std_2, trajectory_std]
            confidence_2 = np.r_[confidence_2, conf_interval]
        elif num_steps == 3:
            error_3 = np.r_[error_3, sq_errors]
            mean_3 = np.r_[mean_3, trajectory_mean]
            std_3 = np.r_[std_3, trajectory_std]
            confidence_3 = np.r_[confidence_3, conf_interval]


        # plots TODO: clean up more and add other relevant plots
        # xt_vals = np.array(xt_vals).reshape(td_agent.num_steps, td_agent.state_dim)
        # zt_vals = np.array(zt_vals)
        # plotting.plot_kde(samples_np,None,xt_vals,zt_vals)
        # plotting.plot_dist(samples_np,None)
        ##pose_plot = sbs.kdeplot(samples_np[:,0], color='purple')
        # sbs.distplot(samples_np[:,0], color='purple')
        ##sbs.kdeplot(samples_np[:,1], color='red')
        # sbs.distplot(samples_np[:,1], color='red', kde_kws={'bw': 2.0})
        # sbs.distplot(samples_np[:,2], color='green')
        ##sbs.kdeplot(samples_np[:,2], color='green')
        # sbs.distplot(samples_np[:,3], color='blue')
        ##sbs.kdeplot(samples_np[:,3], color='blue')
        
        pose_plot = sbs.distplot(samples_np[:,0], color='purple', norm_hist=True, kde_kws={'bw': 'scott'})
        sbs.distplot(samples_np[:,1], color='red', kde_kws={'bw': 'scott'})
        sbs.distplot(samples_np[:,2], color='green', kde_kws={'bw': 'scott'})
        sbs.distplot(samples_np[:,3], color='blue', kde_kws={'bw': 'scott'})
        
        #plt.show()

        true_scale = tf.sqrt(meas_var + lm1_prior_var).eval(session=sess)
        xs = np.linspace(-2.0, 8.0, 100)
        orig_xs, orig_ps = pose_plot.get_lines()[0].get_data()
        ps = spi.interp1d(orig_xs, orig_ps,fill_value=np.array(1e-10), bounds_error=False)(xs)

        qs = (1./3.)*sps.norm.pdf(xs, loc=0.0, scale=true_scale) + \
             (1./3.)*sps.norm.pdf(xs, loc=2.0, scale=true_scale) + \
             (1./3.)*sps.norm.pdf(xs, loc=6.0, scale=true_scale)
        ##plt.plot(xs, ps)
        ##plt.plot(xs, qs.ravel(), color="red")
        ##plt.show()
        approx_kl = metrics.kld(qs, ps, support=(np.max(xs) - np.min(xs)))
        approx_kl = sess.run([approx_kl])
        print("Approx KL: ", approx_kl)
        all_kls.append(approx_kl)
    if num_steps == 1:
        with open('output/BPFerror_1.csv', mode = 'w') as file:
            file_writer = csv.writer(file, delimiter = ',')
            file_writer.writerow(['robot', 'landmark 1', 'landmark 2', 'landmark 3'])
            for i in range(num_seeds):
                j = i*4
                file_writer.writerow(error_1[j:j+4])
        with open('output/BPFmean_1.csv', mode = 'w') as file:
            file_writer = csv.writer(file, delimiter = ',')
            file_writer.writerow(['robot', 'landmark 1', 'landmark 2', 'landmark 3'])
            for i in range(num_seeds):
                j = i*4
                file_writer.writerow(mean_1[j:j+4])
        with open('output/BPFstd_1.csv', mode = 'w') as file:
            file_writer = csv.writer(file, delimiter = ',')
            file_writer.writerow(['robot', 'landmark 1', 'landmark 2', 'landmark 3'])
            for i in range(num_seeds):
                j = i*4
                file_writer.writerow(std_1[j:j+4])
        with open('output/BPFconfidence_1.csv', mode = 'w') as file:
            file_writer = csv.writer(file, delimiter = ',')
            file_writer.writerow(['robot lower', 'robot upper', 'landmark 1 lower', 'landmark 1 upper', 'landmark 2 lower', 'landmark 2 upper', 'landmark 3 lower', 'landmark 3 upper'])
            for i in range(num_seeds):
                j = i*8
                file_writer.writerow(confidence_1[j:j+8])
    elif num_steps == 2:
        with open('output/BPFerror_2.csv', mode = 'w') as file:
            file_writer = csv.writer(file, delimiter = ',')
            file_writer.writerow(['robot', 'landmark 1', 'landmark 2', 'landmark 3'])
            for i in range(num_seeds):
                j = i*4
                file_writer.writerow(error_2[j:j+4])
        with open('output/BPFmean_2.csv', mode = 'w') as file:
            file_writer = csv.writer(file, delimiter = ',')
            file_writer.writerow(['robot', 'landmark 1', 'landmark 2', 'landmark 3'])
            for i in range(num_seeds):
                j = i*4
                file_writer.writerow(mean_2[j:j+4])
        with open('output/BPFstd_2.csv', mode = 'w') as file:
            file_writer = csv.writer(file, delimiter = ',')
            file_writer.writerow(['robot', 'landmark 1', 'landmark 2', 'landmark 3'])
            for i in range(num_seeds):
                j = i*4
                file_writer.writerow(std_2[j:j+4])
        with open('output/BPFconfidence_2.csv', mode = 'w') as file:
            file_writer = csv.writer(file, delimiter = ',')
            file_writer.writerow(['robot lower', 'robot upper', 'landmark 1 lower', 'landmark 1 upper', 'landmark 2 lower', 'landmark 2 upper', 'landmark 3 lower', 'landmark 3 upper'])
            for i in range(num_seeds):
                j = i*8
                file_writer.writerow(confidence_2[j:j+8])
    elif num_steps == 3:
        with open('output/BPFerror_3.csv', mode = 'w') as file:
            file_writer = csv.writer(file, delimiter = ',')
            file_writer.writerow(['robot', 'landmark 1', 'landmark 2', 'landmark 3'])
            for i in range(num_seeds):
                j = i*4
                file_writer.writerow(error_3[j:j+4])
        with open('output/BPFmean_3.csv', mode = 'w') as file:
            file_writer = csv.writer(file, delimiter = ',')
            file_writer.writerow(['robot', 'landmark 1', 'landmark 2', 'landmark 3'])
            for i in range(num_seeds):
                j = i*4
                file_writer.writerow(mean_3[j:j+4])
        with open('output/BPFstd_3.csv', mode = 'w') as file:
            file_writer = csv.writer(file, delimiter = ',')
            file_writer.writerow(['robot', 'landmark 1', 'landmark 2', 'landmark 3'])
            for i in range(num_seeds):
                j = i*4
                file_writer.writerow(std_3[j:j+4])
        with open('output/BPFconfidence_3.csv', mode = 'w') as file:
            file_writer = csv.writer(file, delimiter = ',')
            file_writer.writerow(['robot lower', 'robot upper', 'landmark 1 lower', 'landmark 1 upper', 'landmark 2 lower', 'landmark 2 upper', 'landmark 3 lower', 'landmark 3 upper'])
            for i in range(num_seeds):
                j = i*8
                file_writer.writerow(confidence_3[j:j+8])
    print("Avg KL: ", np.mean(all_kls))
    print("KL std: ", np.std(all_kls))


