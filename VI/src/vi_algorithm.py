import numpy as np
from scipy import stats, special

import vi_statistic as vis

def calc_r(outcomes, dirichlet_param, beta_param):
   ro_nk = vis.calc_ro(outcomes, dirichlet_param, beta_param)
   sum_ro_nk = np.sum(ro_nk, axis=1).reshape(ro_nk.shape[0], 1)
   return ro_nk/sum_ro_nk

def update_dirichlet_param(dirichlet_param, r_nk):
   param_new = dirichlet_param + np.sum(r_nk,axis=0)  
   return param_new

def update_beta_param2(beta_param, r_nk, outcomes):
   heads = outcomes[:,0].reshape(outcomes.shape[0],1)
   tails = outcomes[:,1].reshape(outcomes.shape[0],1)
   a_update = np.sum(heads*r_nk, axis=0)
   b_update = np.sum(tails*r_nk, axis=0)
   beta_update = np.stack((a_update, b_update)).reshape(beta_param.shape)
   return beta_param + beta_update

def ELBO(r_nk, dirichlet_old, beta_old, dirichlet_new, beta_new, experiments):
   elbo = vis.exp_p_X(r_nk, beta_old, experiments) + vis.exp_p_Z(r_nk, dirichlet_old) + \
          vis.exp_p_pi(dirichlet_old) + vis.exp_p_theta(beta_old) - \
          vis.exp_q_Z(r_nk) - vis.exp_p_theta(beta_new) - vis.exp_p_pi(dirichlet_new)
   #print("p_X: ", exp_p_X(r_nk, beta_old, experiments))
   #print("p_Z: ", exp_p_Z(r_nk, dirichlet_old))
   #print("q_Z: ", exp_q_Z(r_nk))
   #print("pi_old: ", exp_p_pi(dirichlet_old))
   #print("pi_new: ", exp_p_pi(dirichlet_new))
   #print("theta_old: ", exp_p_theta(beta_old))
   #print("theta_new: ", exp_p_theta(beta_new))
   #print("elbo: ", float(elbo))
   return float(elbo[0])

