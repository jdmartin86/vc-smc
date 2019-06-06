import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sbs

class TwoDoorsAgent():
    def __init__(self,
                 num_steps=1,
                 state_dim=1,
                 num_landmarks=0,
                 landmark_dim=1,
                 latent_dim=None,
                 observ_dim=1):
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
        # Target model params
        self.target_params = tf.placeholder(dtype=tf.float32,shape=(10,1))

    def sim_target(self, num_samples):
        mean_choice = np.random.choice(a=[0.0, 2.0], p=[0.7, 0.3], size=num_samples)
        samples = [tf.random_normal(shape=(1,self.latent_dim), mean=mc, stddev=0.5) for mc in mean_choice]
        return tf.reshape(tf.convert_to_tensor(samples, dtype=tf.float32), [num_samples, self.latent_dim])

    def sim_proposal(self, t, x_prev, observ, num_particles, proposal_params):
        if t > 0:
            return x_prev + tf.random_normal(shape=(num_particles, self.latent_dim), dtype=tf.float32)
        else:
            return tf.random_normal(shape=(num_particles, self.latent_dim))

    def log_proposal_copula(self, t, x_curr, x_prev, observ):
        num_particles = x_curr.get_shape().as_list()[0]
        return tf.zeros(shape=(num_particles), dtype=tf.float32)

    def log_normal(self, x, mu, Sigma):
        dim = Sigma.shape[0]
        sign, logdet = np.linalg.slogdet(Sigma)
        log_norm = -0.5*dim*np.log(2.*np.pi) - 0.5*logdet
        Prec = np.linalg.inv(Sigma)
        ls_term = -0.5*tf.reduce_sum((x-mu)*Prec*(x-mu).T,1)
        # return log_norm - 0.5*tf.reduce_sum((x-mu)*tf.tensordot(Prec,(x-mu).T,1).T)
        return tf.convert_to_tensor(log_norm, dtype=tf.float64) + ls_term

    def log_mixture(self, x, y, Sigma, p1=0.7, p2=0.3, mu1=0.0, mu2=2.0):
        return tf.log(p1*tf.exp(self.log_normal(x, mu1, Sigma)) +
                      p2*tf.exp(self.log_normal(x, mu2, Sigma)))

    def log_target(self, t, x_curr, x_prev, observ):
        logG = self.log_mixture(x_curr, 0.0, 0.5*np.eye(1))

    def log_proposal_marginal(self, t, x_curr, x_prev, observ):
        return self.log_normal(x, 0.0, 1.0)

    def log_proposal(self, t, x_curr, x_prev, observ, proposal_params):
        return self.log_proposal_copula(t, x_curr, x_prev, observ) + \
               self.log_proposal_marginal(t, x_curr, x_prev, observ)


if __name__ == '__main__':
    num_particles = 1000
    proposal_params = np.zeros(10)
    td_agent = TwoDoorsAgent()
    sess = tf.Session()
    samps = td_agent.sim_proposal(0, None, None, num_particles, proposal_params)
    samp_values = samps.eval(session=sess)
    print(samp_values)
    sbs.distplot(samp_values, color='red')

    target_samples = td_agent.sim_target(num_particles)
    target_sample_values = target_samples.eval(session=sess)
    print(target_sample_values)
    print(samp_values.shape)
    print(target_sample_values.shape)
    sbs.distplot(target_sample_values, color='green')
    # plt.show()

    query_points = np.linspace(-2.0, 4.0, 50)
    print("QP shape", query_points.shape)
    query_values = np.array([tf.exp(td_agent.log_mixture(xi, 0.0, 0.25*np.eye(1))).eval(session=sess) for xi in query_points]).ravel()
    plt.plot(query_points, query_values, color='blue')
    plt.show()
    # target_values = np.array([tf.exp(td_agent.log_target(1, np.array([[xi]]), xi, observ=0.0)).eval(session=sess) for xi in query_points]).ravel()

