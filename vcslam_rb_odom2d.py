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
                 rs=np.random.RandomState(0)):
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
        if not latent_dim:
            self.latent_dim = self.num_steps * (self.state_dim + self.landmark_dim*self.num_landmarks)
        else:
            self.latent_dim = latent_dim
        # Observation dimensionality (direct observations of x_door)
        self.observ_dim = observ_dim
        # Proposal params
        self.proposal_params = tf.placeholder(dtype=tf.float32,shape=(10,1))

        self.rs = rs

    def get_dependency_param_shape(self):
        return 0

    def get_marginal_param_shape(self):
        return [self.num_steps, self.state_dim*3]

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

    def sim_proposal(self, t, x_prev, observ, proposal_params):
        init_pose, init_cov, A, Q, C, R = self.target_params
        num_particles = x_prev.get_shape().as_list()[0]
        proposal_marg_params = proposal_params[1]
        mut = proposal_marg_params[t,0:3]
        lint = proposal_marg_params[t,3:6]
        log_s2t = proposal_marg_params[t,6:9]
        s2t = tf.exp(log_s2t)
        if t > 0:
            mu = mut + tf.matmul(A, x_prev)*lint
        else:
            mu = mut + lint*tf.zeros([3,])#init_pose
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
            logF = self.log_normal(x_curr, tf.matmul(A, x_prev), Q)
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
    init_cov = 0.25*tf.eye(3,3,dtype=np.float32)
    A = 1.1*tf.eye(3,3,dtype=np.float32)
    Q = 0.5*tf.eye(3,3,dtype=np.float32)
    C = tf.eye(2,3,dtype=np.float32)
    R = 0.25*tf.eye(2,2,dtype=np.float32)
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
    print type(zt_vals)
    # print("X True vals: ", xt_vals)
    # print("Z True vals: ", zt_vals)

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
    num_train_steps = 10000
    # Learning rate for the distribution
    lr_m = 0.005
    # Random seed for VCSLAM
    slam_rs = np.random.RandomState(0)

    # Create the VCSLAM instance with above parameters
    vcs = VCSLAM(vcs_agent = td_agent,
                 observ = zt_vals,
                 num_particles = num_particles,
                 num_train_steps = num_train_steps,
                 lr_m = lr_m,
                 rs = slam_rs)
    # tf.reset_default_graph()
    # td_agent = RangeBearingAgent(target_params=target_params, rs=agent_rs, num_steps=num_steps)
    opt_propsal_params, train_sess = vcs.train(vcs_agent = td_agent)
    num_samps = 100
    my_vars = [vcs.sim_q(opt_propsal_params, target_params, zt_vals, td_agent) for i in range(num_samps)]
    my_samples = train_sess.run(my_vars)
    print("done1")
    samples_np = np.array(my_samples)
    plt.scatter(samples_np[:,0], samples_np[:,1], color='blue')
    # sbs.kdeplot(samples_np[:,0], samples_np[:,1])

    xt_vals = np.array(xt_vals).reshape(td_agent.num_steps, td_agent.state_dim)
    # gen_sample_values = np.array([sess.run([td_agent.generate_data()[0]]) for i in range(num_samps)]).reshape(num_samps, td_agent.state_dim)
    gen_vars = [td_agent.generate_data()[0] for i in range(num_samps)]
    gen_sample_values = np.array(sess.run(gen_vars)).reshape(num_samps, td_agent.state_dim)

    print(gen_sample_values.shape)
    print(xt_vals)
    plt.scatter(xt_vals[:,0], xt_vals[:,1], color='red')
    # sbs.kdeplot(gen_sample_values[:,0], gen_sample_values[:,1], color='red')
    plt.scatter(gen_sample_values[:,0], gen_sample_values[:,1], color='black')
    plt.show()

    # slam_rs = np.random.RandomState(0)
    # observ = np.array([0.0])
    # vcs = VCSLAM(vcs_agent = td_agent, observ = observ, num_particles = num_particles, num_train_steps=1000, lr_m=0.001, rs=slam_rs)
    # # tf.get_default_graph().finalize()
    # opt_proposal_params, sess = vcs.train(vcs_agent = td_agent)

    # print(sess.run(opt_proposal_params))

    # num_samps = 200
    # my_vars = [vcs.sim_q(opt_proposal_params, None, observ, td_agent) for i in range(num_samps)]
    # my_samples = sess.run(my_vars)
    # print(my_samples[0])
    # # print(my_samples)
    # sbs.distplot(my_samples, color='blue')

    # # uncomment to do plotting of log target
    # # target_samples = td_agent.sim_target(num_particles)
    # # target_sample_values = target_samples.eval(session=sess)
    # # print(target_sample_values)
    # # print(target_sample_values.shape)
    # # sbs.distplot(target_sample_values, color='green')
    # # plt.show()

    # query_points = np.linspace(-2.0, 4.0, 50)
    # print("QP shape", query_points.shape)
    # # query_values = np.array([tf.exp(td_agent.log_mixture(xi, 0.0, 0.25*np.eye(1))).eval(session=sess) for xi in query_points]).ravel()
    # target_vars = [tf.exp(td_agent.log_target(1, tf.constant([[xi]],dtype=tf.float32), xi, observ=0.0)) for xi in query_points]
    # target_values = np.array(sess.run([target_vars])).ravel()
    # plt.plot(query_points, target_values, color='red')
    # plt.show()

