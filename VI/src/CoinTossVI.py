import numpy as np
import operator
import matplotlib.pyplot as plt
from scipy import stats, special

import pyprint
#import sympy as sp

import vi_algorithm as via

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
      r_nk = via.calc_r(experiments, dirichlet_param, beta_param)
      #print("ro_nk: ", ro_nk)
      #r_nk = via.calc_r(ro_nk)
      #print("r_nk: ", r_nk)
      dirichlet_new = via.update_dirichlet_param(dirichlet_param, r_nk)
      print("dirichlet_new: ", dirichlet_new)
      beta_new = via.update_beta_param2(beta_param, r_nk, experiments)
      print("beta_new: ", beta_new)
      elbo_new = via.ELBO(r_nk, dirichlet_param, beta_param, dirichlet_new, beta_new, experiments)
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
