import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sbs
from vcsmc_marginal_only import *
from tensorflow.python.client import device_lib
import tensorflow.contrib.distributions as tfd

# Remove warnings
tf.logging.set_verbosity(tf.logging.ERROR)

class RangeBearingAgent(VCSLAMAgent):
    def __init__(self,
                 target_params,
                 num_steps=2,
                 state_dim=3,
                 num_landmarks=0,
                 landmark_dim=2,
                 latent_dim=None,
                 observ_dim=2,
                 rs=np.random.RandomState(0),
                 prop_scale=0.5):
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

        self.rs = rs

    def get_dependency_param_shape(self):
        return 0

    def get_marginal_param_shape(self):
        return [self.num_steps, self.state_dim*3]

    def init_marg_params(self):
        T = self.num_steps
        Dx = self.state_dim
        marg_params = np.array([np.array([self.prop_scale * self.rs.randn(Dx), # Bias
                 1. + self.prop_scale * self.rs.randn(Dx), # Linear times A/mu0
                 self.prop_scale * self.rs.randn(Dx)]).ravel() # Log-var
                for t in range(T)])
        print( "Marg param shape: ", marg_params.shape )
        return marg_params

    def generate_data(self):
        # print(self.target_params)
        init_pose, init_cov, A, Q, C, R = self.target_params
        Dx = init_pose.get_shape().as_list()[0]
        Dz = R.get_shape().as_list()[0]

        x_true = []
        z_true = []

        for t in range(self.num_steps):
            if t > 0:
                x_true.append(tf.transpose(tfd.MultivariateNormalFullCovariance(loc=tf.transpose(tf.matmul(A,x_true[t-1])),
                                                                   covariance_matrix=Q).sample(seed=self.rs.randint(0,1234))))
            else:
                x_sample = tf.transpose(tfd.MultivariateNormalFullCovariance(loc=tf.transpose(init_pose),
                                                                   covariance_matrix=init_cov).sample(seed=self.rs.randint(0,1234)))
                x_true.append(x_sample)
                # x_true.append(tfd.MultivariateNormalFullCovariance(loc=init_pose,
                #                                                    covariance_matrix=init_cov).sample(seed=self.rs.randint(0,1234)))
            z_true.append(tf.transpose(tfd.MultivariateNormalFullCovariance(loc=tf.transpose(tf.matmul(C,x_true[t])),
                                                               covariance_matrix=R).sample(seed=self.rs.randint(0,1234))))
        return x_true, z_true

    # def log_marginal_likelihood(self, observ):
    #     """
    #         Completely untested
    #     """
    #     init_pose, init_cov, A, Q, C, R = self.target_params
    #     Dx = init_pose.get_shape().as_list()[0]
    #     Dy = R.get_shape().as_list()[0]
    #     log_likelihood = 0.0
    #     xfilt = tf.zeros(Dx)
    #     Pfilt = tf.zeros([Dx, Dx])
    #     xpred = init_pose
    #     Ppred = init_cov
    #     for t in range(self.num_steps):
    #         if t > 0:
    #             # Predict Step
    #             xpred = tf.matmul(A, xfilt)
    #             Ppred = tf.matmul(A, tf.matmul(Pfilt, tf.transpose(A))) + Q
    #         # Update step
    #         yt = observ[t,:] - tf.matmul(C, xpred)
    #         S = tf.matmul(C, tf.matmul(Ppred, tf.transpose(C))) + R
    #         K = tf.transpose(tf.linalg.solve(S, tf.matmul(C, Ppred)))
    #         xfilt = xpred + tf.matmul(K,yt)
    #         Pfilt = Ppred + tf.matmul(K, tf.matmul(C,Ppred))
    #         sign, logdet = tf.linalg.slogdet(S)
    #         log_likelihood += -0.5*(tf.reduce_sum(yt*tf.linalg.solve(S,yt))) + logdet + Dy*tf.log(2.*np.pi)
    #     return log_likelihood


    def lgss_posterior_params(self, observ, T):
        """
            Apply a Kalman filter to the linear Gaussian state space model
            Returns p(x_T | z_{1:T}) when supplied with z's and T
            Completely untested
            I'm a little worried this is wrong
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
                xpred = tf.matmul(A, xfilt)
                Ppred = tf.matmul(A, tf.matmul(Pfilt, tf.transpose(A))) + Q
            # Update step
            yt = observ[t] - tf.matmul(C, xpred)
            S = tf.matmul(C, tf.matmul(Ppred, tf.transpose(C))) + R
            K = tf.transpose(tf.linalg.solve(S, tf.matmul(C, Ppred)))
            xfilt = xpred + tf.matmul(K,yt)
            Pfilt = Ppred + tf.matmul(K, tf.matmul(C,Ppred))
        return xfilt, Pfilt

    # def sim_target(self, t, x_curr, x_prev, observ)
    #     init_pose, init_cov, A, Q, C, R = self.target_params
    #     if t > 0:
    #         logF = self.log_normal(x_curr, tf.matmul(A, x_prev), Q)
    #     else:
    #         logF = self.log_normal(x_curr, init_pose, init_cov)
    #     logG = self.log_normal(tf.transpose(tf.matmul(C, tf.transpose(x_curr))), tf.convert_to_tensor(observ[t], dtype=tf.float32), R)
    #     return logF + logG


    def sim_proposal(self, t, x_prev, observ, proposal_params):
        init_pose, init_cov, A, Q, C, R = self.target_params
        num_particles = x_prev.get_shape().as_list()[0]
        proposal_marg_params = proposal_params[1]
        mut = proposal_marg_params[t,0:3]
        lint = proposal_marg_params[t,3:6]
        log_s2t = proposal_marg_params[t,6:9]
        s2t = tf.exp(log_s2t)
        if t > 0:
            print("xprev shape: ", x_prev.get_shape().as_list())
            print("shape of mult: ", (tf.matmul(A, tf.transpose(x_prev))))
            mu = mut + tf.transpose(tf.matmul(A, tf.transpose(x_prev)))*lint
            print("T1", mu.shape)
        else:
            print("lint: ", lint.get_shape().as_list())
            print("init pose: ", init_pose.get_shape().as_list())
            print("Mut: ", mut.get_shape().as_list())
            mu = mut + lint*tf.reshape(init_pose, (self.state_dim,))
            print("T0", mu.shape)
        # print("x prev shape", x_prev.get_shape().as_list())
        sample = mu + tf.random.normal(x_prev.get_shape().as_list(),seed=self.rs.randint(0,1234))*tf.sqrt(s2t)
        return sample

    def log_proposal_copula(self, t, x_curr, x_prev, observ):
        num_particles = x_curr.get_shape().as_list()[0]
        return tf.zeros(shape=(num_particles), dtype=tf.float32)

    def log_normal(self, x, mu, Sigma):
        dim = Sigma.get_shape().as_list()[0]
        sign, logdet = tf.linalg.slogdet(Sigma)
        log_norm = -0.5*dim*np.log(2.*np.pi) - 0.5*logdet
        Prec = tf.dtypes.cast(tf.linalg.inv(Sigma), dtype=tf.float32)
        # print(mu)
        # print(mu.get_shape().as_list()[0])
        first_term = x - tf.transpose(mu)
        # print(first_term.get_shape().as_list())
        second_term = tf.transpose(tf.matmul(Prec, tf.transpose(x-tf.transpose(mu))))
        ls_term = -0.5*tf.reduce_sum(first_term*second_term,1)
        # print(ls_term.get_shape().as_list())
        return tf.cast(log_norm, dtype=tf.float32) + tf.cast(ls_term, dtype=tf.float32)

    # def log_mixture(self, x, y, Sigma, p1=0.7, p2=0.3, mu1=0.0, mu2=2.0):
    #     return tf.log(p1*tf.exp(self.log_normal(x, mu1, Sigma)) +
    #                   p2*tf.exp(self.log_normal(x, mu2, Sigma)))

    def log_target(self, t, x_curr, x_prev, observ):
        init_pose, init_cov, A, Q, C, R = self.target_params
        if t > 0:
            logF = self.log_normal(x_curr, tf.matmul(A, tf.transpose(x_prev)), Q)
        else:
            logF = self.log_normal(x_curr, init_pose, init_cov)
        logG = self.log_normal(tf.transpose(tf.matmul(C, tf.transpose(x_curr))), tf.convert_to_tensor(observ[t], dtype=tf.float32), R)
        return logF + logG

    def log_proposal_marginal(self, t, x_curr, x_prev, observ, proposal_params):
        init_pose, init_cov, A, Q, C, R = self.target_params
        proposal_marg_params = proposal_params[1]
        mut = proposal_marg_params[t,0:3]
        lint = proposal_marg_params[t,3:6]
        log_s2t = proposal_marg_params[t,6:9]
        s2t = tf.exp(log_s2t)
        if t > 0:
            mu = mut + tf.matmul(A, x_prev)
        else:
            mu = tf.transpose(mut + lint*tf.transpose(init_pose))
        return self.log_normal(x_curr, mu, tf.diag(s2t))

    def log_proposal(self, t, x_curr, x_prev, observ, proposal_params):
        return self.log_proposal_copula(t, x_curr, x_prev, observ) + \
               self.log_proposal_marginal(t, x_curr, x_prev, observ, proposal_params)

    def log_weights(self, t, x_curr, x_prev, observ, proposal_params):
        target_log = self.log_target(t, x_curr, x_prev, observ)
        target_log = tf.debugging.check_numerics(target_log, "Target log error")
        prop_log = self.log_proposal(t, x_curr, x_prev, observ, proposal_params)
        prop_log = tf.debugging.check_numerics(prop_log, "Proposal log error")
        return target_log - prop_log

if __name__ == '__main__':
    # List available devices
    local_device_protos = device_lib.list_local_devices()
    print([x.name for x in local_device_protos])
    # Optionally use accelerated computation
    # with tf.device("/device:XLA_CPU:0"):
    # Set random seeds
    np.random.seed(1)
    tf.random.set_random_seed(1)
    agent_rs = np.random.RandomState(0)

    # Number of steps for the trajectory
    num_steps = 1
    # True target parameters
    # Consider replacing this with "map", "initial_pose", "true_measurement_model", and "true_odometry_model"
    init_pose = tf.zeros([3,1],dtype=np.float32)
    init_cov = tf.eye(3,3,dtype=np.float32)
    A = 1.1*tf.eye(3,3,dtype=np.float32)
    Q = 0.5*tf.eye(3,3,dtype=np.float32)
    C = tf.eye(2,3,dtype=np.float32)
    R = tf.eye(2,2,dtype=np.float32)
    target_params = [init_pose,
                     init_cov,
                     A,
                     Q,
                     C,
                     R]
    td_agent = RangeBearingAgent(target_params=target_params, rs=agent_rs, num_steps=num_steps)
    x_true, z_true = td_agent.generate_data()
    sess = tf.Session()
    xt_vals, zt_vals = sess.run([x_true, z_true])
    print( type(zt_vals) )
    # Number of samples to use for plotting
    # print("X True vals: ", xt_vals)
    print("Z True vals: ", zt_vals[0].shape)

    """
        Plot True
    """
    # xt_vals = np.array(xt_vals).reshape(td_agent.num_steps, td_agent.state_dim)
    # # print("Shape of X true", xt_vals.shape)

    # points = np.array([xt_vals[:,0], xt_vals[:,1]]).transpose().reshape(-1,1,2)
    # # print points.shape  # Out: (len(x),1,2)

    # segs = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
    # # print segs.shape
    # color = np.linspace(0,1,xt_vals.shape[0])
    # lc = LineCollection(segs, cmap=plt.get_cmap('viridis'))
    # lc.set_array(color)
    # plt.gca().add_collection(lc)
    # plt.xlim(xt_vals[:,0].min(), xt_vals[:,0].max())
    # plt.ylim(xt_vals[:,1].min(), xt_vals[:,1].max())
    # plt.plot(xt_vals[:,0], xt_vals[:,1])
    # plt.show()


    # Number of particles to use
    num_particles = 100
    # Number of iterations to fit the proposal parameters
    num_train_steps = 1000
    # Learning rate for the distribution
    lr_m = 0.001
    # Random seed for VCSLAM
    slam_rs = np.random.RandomState(0)

    # Number of samples to use for plotting
    num_samps = 10000

    # Create the VCSLAM instance with above parameters
    vcs = VCSLAM(vcs_agent = td_agent,
                 observ = zt_vals,
                 num_particles = num_particles,
                 num_train_steps = num_train_steps,
                 lr_m = lr_m,
                 rs = slam_rs)

    # Get posterior samples (since everything is linear Gaussian, just do Kalman filtering)
    post_mean, post_cov = td_agent.lgss_posterior_params(zt_vals, 1)
    print("Post mean shape: ", post_mean.get_shape().as_list())
    # post_samples = tfd.MultivariateNormalFullCovariance(loc=tf.transpose(post_mean),
    #                                                     covariance_matrix=post_cov).sample(sample_shape = num_samps, seed = td_agent.rs.randint(0,1234))
    # post_values = sess.run([post_samples])
    p_mu, p_cov = sess.run([post_mean, post_cov])
    print(p_mu.T.shape)
    post_values = td_agent.rs.multivariate_normal(mean=p_mu.ravel(), cov=p_cov, size=num_samps)
    print("pv shape: ", post_values.shape)
    post_values = np.array(post_values).reshape((num_samps, td_agent.state_dim))
    print("Post values shape: ", np.array(post_values).shape)
    sbs.kdeplot(post_values[:,0], post_values[:,1], color='green')

    opt_propsal_params, train_sess = vcs.train(vcs_agent = td_agent)
    opt_propsal_params = train_sess.run(opt_propsal_params)
    print("opt params", opt_propsal_params)
    my_vars = [vcs.sim_q(opt_propsal_params, target_params, zt_vals, td_agent)]
    my_samples = [train_sess.run(my_vars) for i in range(num_samps)]
    print("done1")
    samples_np = np.array(my_samples).reshape(num_samps, td_agent.state_dim)
    print(samples_np.shape)
    # plt.scatter(samples_np[:,0], samples_np[:,1], color='blue')
    sbs.kdeplot(samples_np[:,0], samples_np[:,1], color='blue')

    xt_vals = np.array(xt_vals).reshape(td_agent.num_steps, td_agent.state_dim)
    # gen_sample_values = np.array([sess.run([td_agent.generate_data()[0]]) for i in range(num_samps)]).reshape(num_samps, td_agent.state_dim)
    # gen_vars = [td_agent.generate_data()[0] for i in range(num_samps)]
    # gen_sample_values = np.array(sess.run(gen_vars)).reshape(num_samps, td_agent.state_dim)

    # print(gen_sample_values.shape)
    print(xt_vals)
    zt_vals = np.array(zt_vals)
    plt.scatter(xt_vals[:,0], xt_vals[:,1], color='red')
    plt.figure()
    sbs.kdeplot(samples_np[:,1],samples_np[:,2],color='blue')
    sbs.kdeplot(post_values[:,1],post_values[:,2], color='green')
    plt.figure()
    sbs.distplot(samples_np[:,1], color='blue')
    sbs.distplot(post_values[:,1], color='green')
    # plt.scatter(zt_vals[:,0], zt_vals[:,1], color='green')
    # sbs.kdeplot(gen_sample_values[:,0], gen_sample_values[:,1], color='red')
    # plt.scatter(gen_sample_values[:,0], gen_sample_values[:,1], color='black')

    plt.show()

