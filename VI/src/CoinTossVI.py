import numpy as np
import operator
import matplotlib.pyplot as plt
from scipy import stats, special

import pyprint
#import sympy as sp

#true (unknown theta values)
Theta = [0.8, 0.45]
#true (unknown coin selections)
coins  = ['B', 'A', 'A', 'B', 'A']
#experiment outcomes (5x10)
events = (['HTTTHHTHTH',
          'HHHHTHHHHH',
          'HTHHHHHTHH',
          'HTHTTTHHTT',
          'THHHTHHHTH'])

bin_events = np.array([[int(letter=='H') for letter in word] for word in events])
Total = len(events[0])
heads = np.sum(bin_events, axis = 1)
tails = Total - heads
experiments = np.column_stack((heads, tails))
print("shape", experiments.shape)

dirichlet_param = np.array([[0.501, 0.501]], dtype=np.float64)
r_nk = 2*np.ones((5,2))
#ro_nk = 2*np.ones((5,2))
beta_k1 = np.array([2, 1], dtype=np.float64)
beta_k2 = np.array([3, 5], dtype=np.float64)
beta_param = np.column_stack((np.transpose(beta_k1), np.transpose(beta_k2)))

def calc_ro(outcomes, dirichlet_param, beta_param):
   alpha_tilde = np.sum(dirichlet_param)
   exp_pi = special.psi(dirichlet_param) - special.psi(alpha_tilde)
   heads = outcomes[:,0].reshape(outcomes.shape[0],1)
   tails = outcomes[:,1].reshape(outcomes.shape[0],1)
   beta_a = beta_param[0,:]
   beta_b = beta_param[1,:]
   exp_theta = heads*special.psi(beta_a) + tails*special.psi(beta_b) - (heads+tails)*special.psi(beta_a+beta_b)
   ln_ro_nk = exp_pi + exp_theta
   return np.exp(ln_ro_nk)

def calc_r(ro_nk):
   sum_ro_nk = np.sum(ro_nk, axis=1).reshape(ro_nk.shape[0], 1)
   return ro_nk/sum_ro_nk

def update_dirichlet_param(dirichlet_param, r_nk):
   #print("r_k: ", np.sum(r_nk,axis=0))
   param_new = dirichlet_param + np.sum(r_nk,axis=0)  
   return param_new

def update_beta_param(beta_param, r_nk, outcomes):
   a_update = np.sum(outcomes_t[:,0]*r_nk).reshape(1, beta_param.shape[1])
   b_update = np.sum(outcomes_t[:,1]*r_nk).reshape(1, beta_param.shape[1])
   beta_update = np.stack((a_update, b_update)).reshape(beta_param.shape)
   return beta_param + beta_update
   
def update_beta_param2(beta_param, r_nk, outcomes):
   heads = outcomes[:,0].reshape(outcomes.shape[0],1)
   tails = outcomes[:,1].reshape(outcomes.shape[0],1)
   a_update = np.sum(heads*r_nk, axis=0)
   b_update = np.sum(tails*r_nk, axis=0)
   beta_update = np.stack((a_update, b_update)).reshape(beta_param.shape)
   return beta_param + beta_update

def logC(a):
   return special.gammaln(np.sum(a, axis=1)) - np.sum(special.gammaln(a), axis=1)

def psi_diff(param):
   return special.psi(param) - special.psi(np.sum(param, axis=1)) 

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

def ELBO(r_nk, dirichlet_old, beta_old, dirichlet_new, beta_new):
   elbo = exp_p_X(r_nk, beta_old, experiments) + exp_p_Z(r_nk, dirichlet_old) + exp_p_pi(dirichlet_old) + exp_p_theta(beta_old) - \
          exp_q_Z(r_nk) - exp_p_theta(beta_new) - exp_p_pi(dirichlet_new)
   print("p_X: ", exp_p_X(r_nk, beta_old, experiments))
   print("p_Z: ", exp_p_Z(r_nk, dirichlet_old))
   print("q_Z: ", exp_q_Z(r_nk))
   #print("pi_old: ", exp_p_pi(dirichlet_old))
   #print("pi_new: ", exp_p_pi(dirichlet_new))
   print("theta_old: ", exp_p_theta(beta_old))
   print("theta_new: ", exp_p_theta(beta_new))
   print("elbo: ", float(elbo))
   return float(elbo[0])

#print(exp_p_pi(np.array([2, 2]).reshape(1,2)))
#print(exp_p_theta(np.array([[2, 2],[1, 1]]).reshape(2,2)))
#print(exp_q_Z(r_nk))
#print(exp_p_Z(r_nk, dirichlet_param))
#print(exp_p_X(r_nk, beta_param, experiments))
#print(dirichlet_param)
#print(np.shape(dirichlet_param))
#print(r_nk)
elbo_history = []
elbo_old = 0
if __name__ == '__main__':
   for i in range(10000):
      ro_nk = calc_ro(experiments, dirichlet_param, beta_param)
      #print("ro_nk: ", ro_nk)
      r_nk = calc_r(ro_nk)
      #print("r_nk: ", r_nk)
      dirichlet_new = update_dirichlet_param(dirichlet_param, r_nk)
      print("dirichlet_new: ", dirichlet_new)
      beta_new = update_beta_param2(beta_param, r_nk, experiments)
      print("beta_new: ", beta_new)
      elbo_new = ELBO(r_nk, dirichlet_param, beta_param, dirichlet_new, beta_new)
      print("delta ", i, ": ", elbo_new)
      if(abs(elbo_new - elbo_old) > 0.001):
         elbo_old = elbo_new
         elbo_history.append(elbo_new)
      else:
         break
      dirichlet_param = dirichlet_new
      beta_param = beta_new
      #print("beta", logC(beta_param))

print(beta_param)

#print(elbo_history)
pyprint.plot_beta(beta_param)
pyprint.plot_elbo(elbo_history)
