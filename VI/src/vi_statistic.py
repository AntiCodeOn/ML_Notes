import numpy as np
from scipy import stats, special

def logC(a):
   result = special.gammaln(np.sum(a, axis=1)) - np.sum(special.gammaln(a), axis=1)
   return result.reshape(a.shape[0], 1)

def psi_diff(param):
   return special.psi(param) - special.psi(np.sum(param, axis=1).reshape(param.shape[0], 1)) 

def sum_psi_diff(param):
   return np.sum((param - 1.)*psi_diff(param), axis=1).reshape(param.shape[0], 1)

def exp_p_pi(param):
   return logC(param) + sum_psi_diff(param)

def exp_p_theta(beta_param):
   exp_theta = sum_psi_diff(np.transpose(beta_param))
   return np.sum(exp_theta+logC(np.transpose(beta_param)))

def exp_p_X(r_nk, beta_param, outcomes):
   exp_theta = np.transpose(psi_diff(np.transpose(beta_param)))
   k1_part = np.sum(outcomes * (exp_theta[0, :]).reshape(1, exp_theta.shape[1]), axis=1)
   k2_part = np.sum(outcomes * exp_theta[1, :].reshape(1, exp_theta.shape[1]), axis=1)
   k1_k2 = np.column_stack((k1_part, k2_part))
   return np.sum(r_nk * k1_k2)

def exp_p_Z(r_nk, dirichlet_param):
   return np.sum(r_nk*psi_diff(dirichlet_param))

def exp_q_Z(r_nk):
   return np.sum(r_nk*np.log(r_nk))

def calc_ro(outcomes, dirichlet_param, beta_param):
   exp_pi = psi_diff(dirichlet_param)
   exp_theta = psi_diff(np.transpose(beta_param))
   k1_part = outcomes * exp_theta[0, :].reshape(1, exp_theta.shape[1])
   k2_part = outcomes * exp_theta[1, :].reshape(1, exp_theta.shape[1])
   k1_part_s = np.sum(k1_part, axis=1)
   k2_part_s = np.sum(k2_part, axis=1)
   k1_k2 = np.column_stack((k1_part_s, k2_part_s))
   ln_ro_nk = exp_pi + k1_k2
   return np.exp(ln_ro_nk)
