import numpy as np
import operator
import matplotlib.pyplot as plt
from scipy import stats, special
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

dirichlet_param = np.array([[5, 5]])
r_nk = np.ones((5,2))
ro_nk = 2*np.ones((5,2))
beta_k1 = np.array([2, 2])
beta_k2 = np.array([5, 2])
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
	print("r_k: ", np.sum(r_nk,axis=0))
	param_new = dirichlet_param + np.sum(r_nk,axis=0)	
	return param_new

def update_beta_param(beta_param, r_nk, outcomes):
	r_k = np.sum(r_nk, axis=0).reshape(1,2)
	outcomes_t = np.sum(outcomes, axis=0)
	a_update = outcomes_t[0]*r_k
	b_update = outcomes_t[1]*r_k
	beta_update = np.stack((a_update, b_update)).reshape(beta_param.shape)
	return beta_param + beta_update
	
def update_beta_param2(beta_param, r_nk, outcomes):
	heads = outcomes[:,0].reshape(outcomes.shape[0],1)
	tails = outcomes[:,1].reshape(outcomes.shape[0],1)
	a_update = np.sum(heads*r_nk, axis=0)
	b_update = np.sum(tails*r_nk, axis=0)
	beta_update = np.stack((a_update, b_update)).reshape(beta_param.shape)
	return beta_param + beta_update

#print(dirichlet_param)
#print(np.shape(dirichlet_param))
#print(r_nk)
for i in range(15):
	ro_nk = calc_ro(experiments, dirichlet_param, beta_param)
	print("ro_nk: ", ro_nk)
	r_nk = calc_r(ro_nk)
	print("r_nk: ", r_nk)
	dirichlet_new = update_dirichlet_param(dirichlet_param, r_nk)
	print("dirichlet_new: ", dirichlet_new)
	beta_new = update_beta_param2(beta_param, r_nk, experiments)
	print("beta_new: ", beta_new)
	dirichlet_param = dirichlet_new
	beta_param = beta_new
