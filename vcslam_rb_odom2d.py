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
                 num_landmarks=2,
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
        return [3,1]

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

    # def sim_target(self, num_samples):
    #     mean_choice = self.rs.random.choice(a=[0.0, 2.0], p=[0.7, 0.3], size=num_samples)
    #     samples = [tf.random_normal(shape=(1,self.latent_dim), mean=mc, stddev=0.5, seed=self.rs.randint(0,1234)) for mc in mean_choice]
    #     return tf.reshape(tf.convert_to_tensor(samples, dtype=tf.float32), [num_samples, self.latent_dim])

    def sim_proposal(self, t, x_prev, observ, proposal_params):
        num_particles = x_prev.get_shape().as_list()[0]
        proposal_marg_params = proposal_params[1]
        mut = proposal_marg_params[0]
        lint = proposal_marg_params[1]
        log_s2t = proposal_marg_params[2]
        # mut, lint, log_s2t = proposal_marg_params
        s2t = tf.exp(log_s2t)
        # if t > 0:
        #     return x_prev + tf.random_normal(shape=(num_particles, self.latent_dim), dtype=tf.float32)
        # else:
        #     return tf.random_normal(shape=(num_particles, self.latent_dim))
        mu = mut + lint*0.0
        sample = mu + tf.random.normal((num_particles,),seed=self.rs.randint(0,1234))*tf.sqrt(s2t)
        # print("Sample: ", sample)
        return sample

    def log_proposal_copula(self, t, x_curr, x_prev, observ):
        num_particles = x_curr.get_shape().as_list()[0]
        return tf.zeros(shape=(num_particles), dtype=tf.float32)

    def log_normal(self, x, mu, Sigma):
        dim = 1.0 #Sigma.shape[0]
        sign, logdet = tf.linalg.slogdet(Sigma)
        log_norm = -0.5*dim*tf.log(2.*np.pi) - 0.5*logdet
        Prec = tf.linalg.inv(Sigma)
        first_term = (x-mu)
        second_term = Prec*tf.transpose(x-mu)
        # print("X minus mu", (x-mu).dtype)
        # print("first term shape", (x-mu).shape)
        # print("second_term shape", second_term.shape)
        ls_term = -0.5*tf.reduce_sum(tf.transpose(first_term*second_term),1)
        # ls_term = -0.5*tf.reduce_sum(tf.transpose((x-mu)*tf.tensordot(Prec, tf.transpose(x-mu), 1)),1)
        # return log_norm - 0.5*tf.reduce_sum((x-mu)*tf.tensordot(Prec,(x-mu).T,1).T)
        return tf.convert_to_tensor(log_norm, dtype=tf.float32) + tf.cast(ls_term, dtype=tf.float32)

    def log_mixture(self, x, y, Sigma, p1=0.7, p2=0.3, mu1=0.0, mu2=2.0):
        return tf.log(p1*tf.exp(self.log_normal(x, mu1, Sigma)) +
                      p2*tf.exp(self.log_normal(x, mu2, Sigma)))

    def log_target(self, t, x_curr, x_prev, observ):
        # logF = self.log_normal(x_curr, tf.constant(0.0, dtype=tf.float32), 1.0*tf.eye(1, dtype=tf.float32))
        logG = self.log_mixture(x_curr, tf.constant(0.0, dtype=tf.float32), 0.25*tf.eye(1,dtype=tf.float32))
        # logG = self.log_normal(x_curr, tf.constant(1.0, dtype=tf.float32), 0.25*tf.eye(1,dtype=tf.float32))
        # return logF + logG
        return logG

    def log_proposal_marginal(self, t, x_curr, x_prev, observ, proposal_params):
        proposal_marg_params = proposal_params[1]
        mut = proposal_marg_params[0]
        lint = proposal_marg_params[1]
        log_s2t = proposal_marg_params[2]
        # mut, lint, log_s2t = proposal_marg_params
        s2t = tf.exp(log_s2t)
        # if t > 0:
        #     return x_prev + tf.random_normal(shape=(num_particles, self.latent_dim), dtype=tf.float32)
        # else:
        #     return tf.random_normal(shape=(num_particles, self.latent_dim))
        mu = mut + lint*0.0
        # sample = mu + tf.random.normal((num_particles,),seed=self.rs.randint(0,1234))*tf.sqrt(s2t)
        return self.log_normal(x_curr, mu, s2t*np.eye(1))

    def log_proposal(self, t, x_curr, x_prev, observ, proposal_params):
        return self.log_proposal_copula(t, x_curr, x_prev, observ) + \
               self.log_proposal_marginal(t, x_curr, x_prev, observ, proposal_params)

    def log_weights(self, t, x_curr, x_prev, observ, proposal_params):
        # print(x_curr.get_shape())
        # print(x_curr.dtype)
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
    num_steps = 50
    # True target parameters
    # Consider replacing this with "map", "initial_pose", "true_measurement_model", and "true_odometry_model"
    init_pose = tf.zeros([3,1])
    init_cov = tf.eye(3,3)
    A = 1.1*tf.eye(3,3)
    Q = tf.eye(3,3)
    C = tf.eye(2,3)
    R = tf.eye(2,2)
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
    # print("X True vals: ", xt_vals)
    # print("Z True vals: ", zt_vals)

    xt_vals = np.array(xt_vals).reshape(td_agent.num_steps, td_agent.state_dim)
    # print("Shape of X true", xt_vals.shape)

    points = np.array([xt_vals[:,0], xt_vals[:,1]]).transpose().reshape(-1,1,2)
    # print points.shape  # Out: (len(x),1,2)

    segs = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
    # print segs.shape
    color = np.linspace(0,1,xt_vals.shape[0])
    lc = LineCollection(segs, cmap=plt.get_cmap('viridis'))
    lc.set_array(color)
    plt.gca().add_collection(lc)
    plt.xlim(xt_vals[:,0].min(), xt_vals[:,0].max())
    plt.ylim(xt_vals[:,1].min(), xt_vals[:,1].max())
    # plt.plot(xt_vals[:,0], xt_vals[:,1])
    plt.show()


    # Number of particles to use
    num_particles = 10
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

