"""Experiments for the range-bearing agent using a non-linear transition model.
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorflow.contrib.distributions as tfd
import plotting
import tensorflow_probability as tfp
import correlation_cholesky as cc

from vcsmc import *
import vcslam_agent
import time

import copula_gaussian as cg

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
                 cop_scale=0.5):
        # Target model params
        self.target_params=target_params
        # Number of time steps
        self.num_steps=num_steps
        # Latent state dimensionality (x)
        self.state_dim=state_dim
        # Num landmarks
        self.num_landmarks=num_landmarks
        # Landmark dimensionality (x)
        self.landmark_dim=landmark_dim
        # Latent variable dimensionality
        self.latent_dim=(self.state_dim + self.landmark_dim*self.num_landmarks)
        # Observation dimensionality (direct observations of x_door)
        self.observ_dim=observ_dim
        # Proposal params
        self.prop_scale=prop_scale
        self.cop_scale=cop_scale

        # Random state for sampling
        self.rs=rs

        # initialize dependency models
        self.copula_s = cg.WarpedGaussianCopula(
            loc = tf.zeros(shape=self.state_dim, dtype=tf.float32),
            scale_tril = tf.eye(self.state_dim, dtype=tf.float32),
            marginal_bijectors=[
                cg.NormalCDF(loc=0., scale=1.),
                cg.NormalCDF(loc=0., scale=1.),
                cg.NormalCDF(loc=0., scale=1.)])

    def transition_model(self, x):
        init_pose, init_cov, A, Q, C, R = self.target_params

        y0 = x[0] + 0.1*tf.cos(x[2])
        y1 = x[1] + 0.1*tf.sin(x[2])
        y2 = x[2]
        return tf.concat([y0[None,:],y1[None,:],y2[None,:]],0)

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
        Dx = np.shape(init_pose)[0]
        Dz = np.shape(R)[0]

        
        x_true = []
        z_true = []
        x_true.append(init_pose)
        z = np.matmul(C, init_pose)
        z_true.append(np.array(np.random.multivariate_normal(z.flatten(), R)[:,None], dtype=np.float32))
        for t in range(1,self.num_steps):
            x = x_true[t-1].flatten()
            y0 = x[0] + 0.1*np.cos(x[2])
            y1 = x[1] + 0.1*np.sin(x[2])
            y2 = x[2]
            m = np.array([y0,y1,y2], dtype=np.float32)
            x_true.append(np.array(np.random.multivariate_normal(m, Q)[:,None], dtype=np.float32))
            
            z = np.matmul(C, x_true[t])
            z_true.append(np.array(np.random.multivariate_normal(z.flatten(), R)[:,None], dtype=np.float32))

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
      """Draw from the proposal p( x | x_prev, observ; proposal_params)"""
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
      x1_cdf = cg.NormalCDF(loc=tf.transpose([mu[:,0]]), scale=tf.sqrt(s2t[0]))
      x2_cdf = cg.NormalCDF(loc=tf.transpose([mu[:,1]]), scale=tf.sqrt(s2t[1]))
      x3_cdf = cg.NormalCDF(loc=tf.transpose([mu[:,2]]), scale=tf.sqrt(s2t[2]))
      
      # Build a copula (can also store globally if we want) we would just
      #  have to modify self.copula.scale_tril and
      #  self.copula.marginal_bijectors in each iteration NOTE: I add
      #  tf.eye(3) to L_mat because I think the diagonal has to be > 0
      gc = cg.WarpedGaussianCopula(
          loc=[0., 0., 0.],
          scale_tril=L_mat, 
          marginal_bijectors=[
              x1_cdf,
              x2_cdf,
              x3_cdf])
      return gc.sample(x_prev.get_shape().as_list()[0])

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
        # these are normal CDFs
        x1_cdf = cg.NormalCDF(loc=tf.transpose([mu[:,0]]), scale=tf.sqrt(s2t[0]))
        x2_cdf = cg.NormalCDF(loc=tf.transpose([mu[:,1]]), scale=tf.sqrt(s2t[1]))
        x3_cdf = cg.NormalCDF(loc=tf.transpose([mu[:,2]]), scale=tf.sqrt(s2t[2]))

        # Build a copula (can also store globally if we want) we would just
        #  have to modify self.copula.scale_tril and
        #  self.copula.marginal_bijectors in each iteration NOTE: I add
        #  tf.eye(3) to L_mat because I think the diagonal has to be > 0
        # self.copula_s._bijector = cg.Concat([x1_cdf, x2_cdf, x3_cdf])
        # self.copula_s.distribution.scale_tril = L_mat
        gc = cg.WarpedGaussianCopula(
            loc=[0., 0., 0.],
            scale_tril=L_mat, # TODO This is currently just eye(3), use L_mat!
            marginal_bijectors=[
                x1_cdf,
                x2_cdf,
                x3_cdf])

        return gc.log_prob(x_curr)

    def log_proposal_copula(self,t,x_curr,x_prev,observ,proposal_params):
        """
        Returns the log probability from the copula model described in Equation 4
        """
        return self.log_proposal_copula_s(t,x_curr,x_prev,observ,proposal_params)

    def log_normal(self, x, mu, Sigma):
        dim = np.shape(Sigma)[0]
        sign, logdet = tf.linalg.slogdet(Sigma)
        log_norm = -0.5*dim*np.log(2.*np.pi) - 0.5*logdet
        Prec = tf.dtypes.cast(tf.linalg.inv(Sigma), dtype=tf.float32)
        first_term = x - mu
        second_term = tf.transpose(tf.matmul(Prec, tf.transpose(x-mu)))
        ls_term = -0.5*tf.reduce_sum(first_term*second_term,1)
        return tf.cast(log_norm, dtype=tf.float32) + tf.cast(ls_term, dtype=tf.float32)

    def log_target(self, t, x_curr, x_prev, observ):
        init_pose, init_cov, A, Q, C, R = self.target_params
        if t > 0:
            logF = self.log_normal(x_curr, tf.transpose(self.transition_model(tf.transpose(x_prev))), Q)
        else:
            logF = self.log_normal(x_curr, tf.transpose(init_pose), init_cov)

        logG = self.log_normal(tf.transpose(self.measurement_model(tf.transpose(x_curr))),
                               tf.transpose(tf.convert_to_tensor(observ[t], dtype=tf.float32)), R)
        return logF + logG

    def log_proposal_marginal(self, t, x_curr, x_prev, observ, prop_marg_params):
        """
        In the Gaussian case the multivariate normal with diagonal covariance equals:
        prod_{i=1}^{self.state_dim} p(x_curr[i] | x_prev[i]; proposal_params)
        """
        init_pose, init_cov, A, Q, C, R = self.target_params
        mut = prop_marg_params[t,0:3]
        lint = prop_marg_params[t,3:6]
        s2t = tf.diag_part(Q)
        if t > 0:
            mu = mut + tf.transpose(self.transition_model(tf.transpose(x_prev)))*lint
        else:
            mu = mut + lint*tf.transpose(init_pose)
        log_prob = self.log_normal(x_curr, mu, tf.diag(s2t))
        return log_prob

    def log_proposal(self, t, x_curr, x_prev, observ, proposal_params):
        """
            Note: we don't actually need log_proposal_marginal
            This is because log_proposal_copula computes the whole thing
        """
        prop_copula_params, prop_marg_params = proposal_params
        cl = self.log_proposal_copula(t, x_curr, x_prev, observ, proposal_params)
        return cl

    def log_weights(self, t, x_curr, x_prev, observ, proposal_params):
        target_log = self.log_target(t, x_curr, x_prev, observ)
        prop_log = self.log_proposal(t, x_curr, x_prev, observ, proposal_params)
        return target_log - prop_log

if __name__ == '__main__':
    # Number of steps for the trajectory
    num_steps = 10
    # Number of particles to use during training
    num_train_particles = 100
    # Number of random seeds for experimental trials
    num_seeds = 10
    # Number of samples to use for plotting
    num_samps = 2000
    # Proposal initial scale
    prop_scale = 0.5
    # Copula initial scale
    cop_scale = 0.1
    # Number of EM iterations
    num_train_steps = 0
    # Learning rate for the marginal
    lr_m = 0.01
    # Learning rate for the copula
    lr_d = 0.01

    # True target parameters
    init_pose = np.ones((3,1),dtype=np.float32)
    init_cov = 0.01*np.eye(3,3,dtype=np.float32)
    A = 1.1*np.eye(3,3,dtype=np.float32)
    Q = 0.01*np.eye(3,3,dtype=np.float32)
    C = np.eye(2,3,dtype=np.float32)
    R = 0.01*np.eye(2,2,dtype=np.float32)
    target_params = [init_pose, init_cov, A, Q, C, R]

    # Create the session
    sess = tf.Session()

    # Create the agent
    rs = np.random.RandomState(1)# This remains fixed for the ground truth
    td_agent = RangeBearingAgent(target_params=target_params,
                                 rs=rs,
                                 num_steps=num_steps,
                                 prop_scale=prop_scale,
                                 cop_scale=cop_scale)

    # Generate observations
    trajectory_ref, trajectory_observations = td_agent.generate_data()
    for t in range(num_steps):
        np.savetxt('output/nonlinear/trajectory_ref_{}.txt'.format(t), trajectory_ref[t], delimiter=',')

    # The initial paramters are used for SMC
    initial_dependency_params = td_agent.init_dependency_params()
    initial_marginal_params = td_agent.init_marg_params()
    initial_dependency_params = tf.convert_to_tensor(initial_dependency_params, tf.float32)
    initial_marginal_params = tf.convert_to_tensor(initial_marginal_params, tf.float32)
    initial_proposal_params = [initial_dependency_params, initial_marginal_params]

    trial_mean_errors = []; trial_map_errors = []; trial_means = []
    trial_dep_loss = []; trial_marg_loss = []
    for seed in range(num_seeds):
        start = time.time()
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./logs', sess.graph)
            vcs = VCSLAM(sess = sess,
                         vcs_agent = td_agent,
                         observ = trajectory_observations,
                         num_particles = num_train_particles,
                         num_train_steps = num_train_steps,
                         lr_d = lr_d,
                         lr_m = lr_m,
                         summary_writer = writer)
            
            # Train the model
            est_proposal_params, dep_loss, marg_loss = vcs.train(vcs_agent = td_agent)
            est_proposal_params = sess.run(est_proposal_params)
            est_dep_params, est_marg_params = est_proposal_params
            
            #print("Initial marg params: {}".format(td_agent.init_marg_params().flatten()))
            #print("Estimated marg params: {}".format(est_marg_params.flatten()))
            #print("Initial dep params: {}".format(td_agent.init_dependency_params().flatten()))
            #print("Estimated dep params: {}".format(est_dep_params.flatten()))

            # Sample the model
            particles, map_traj = vcs.sim_q(est_proposal_params,
                                            target_params,
                                            trajectory_observations,
                                            td_agent,
                                            num_samples=num_train_particles)
            particles, map_traj = sess.run([particles, map_traj])
            
        particles = np.squeeze(np.array(particles))
        #print('Estimated trajectory samples: {}'.format(particles[-1,:]))

        # record trial data
        trajectory_ref = np.squeeze(trajectory_ref)
        mean_traj = np.mean(particles, axis=1)

        trial_mean_errors.append(np.sum(np.sqrt(np.sum((trajectory_ref-mean_traj)**2,axis=-1))))
        trial_map_errors.append(np.sum(np.sqrt(np.sum((trajectory_ref-map_traj)**2,axis=-1))))
        trial_dep_loss.append(np.array(dep_loss).flatten())
        trial_marg_loss.append(np.array(marg_loss).flatten())
            
        #save the data
        np.savetxt('output/nonlinear/vcsmc_no_rmse_mean_{}_{}.csv'.format(num_steps,seed), trial_mean_errors, delimiter=',')
        np.savetxt('output/nonlinear/vcsmc_no_rmse_map_{}_{}.csv'.format(num_steps,seed), trial_map_errors, delimiter=',')
        np.savetxt('output/nonlinear/vcsmc_no_dep_loss_{}_{}.csv'.format(num_steps,seed), trial_dep_loss, delimiter=',')
        np.savetxt('output/nonlinear/vcsmc_no_marg_loss_{}_{}.csv'.format(num_steps,seed), trial_marg_loss, delimiter=',')
            
        # Print elapsed time for the trial
        end = time.time()
        graph_vars = [n.name for n in tf.get_default_graph().as_graph_def().node]
        print("Trial #{}: Elapsed time {}, Graph nodes {}".format(seed,
                                                                  end - start,
                                                                  len(graph_vars)))


