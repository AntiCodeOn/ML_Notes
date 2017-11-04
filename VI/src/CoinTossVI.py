import numpy as np
import operator
import matplotlib.pyplot as plt
from scipy import stats, special
#import sympy as sp

#true (unknown theta values)
Theta = [0.8, 0.45]
#true (unknown coin selections)
coins  = ['B', 'A', 'A', 'B', 'A']

def bernoulli(theta, experiment):
	return (theta**experiment[0])*(1-theta)**experiment[1]

def prob_x_theta(theta, experiment):
	return [bernoulli(theta[0], experiment), bernoulli(theta[1], experiment)]

def expect_zk(theta, experiment):
	ex = prob_x_theta(theta, experiment)
	ex_den = (ex[0]+ex[1])
	return [ex[0]/ex_den, ex[1]/ex_den]

def max_likelihood(E, bin_events):
	numTheta0=bin_events*E[:,0:1]
	numTheta1=bin_events*E[:,1:]
	den=10*np.sum(E, axis=0)
	return [np.sum(numTheta0)/den[0], np.sum(numTheta1)/den[1]]

events = (['HTTTHHTHTH',
          'HHHHTHHHHH',
          'HTHHHHHTHH',
          'HTHTTTHHTT',
          'THHHTHHHTH'])

bin_events = np.array([[int(letter=='H') for letter in word] for word in events])
Total = len(events[0])

heads = np.array([event.count('H') for event in events])
tails = Total - heads
experiments = np.column_stack((heads, tails))

dirichlet_param = np.array([[0.5, 0.5]])
r_nk = np.ones((5,2))
ro_nk = 2*np.ones((5,2))
#beta_param 

def calc_ro(outcomes, dirichlet_param, beta_param):
	alpha_tilde = np.sum(dirichlet_param)
	exp_pi = special.digamma(dirichlet_param) - special.digamma(alpha_tilde)
	heads = outcomes[:][0]
	tails = outcomes[:][1]
	beta_a = beta_param[0][:]
	beta_b = beta_param[1][:]
	exp_theta = heads*special.digama(beta_a) + tails*special.digamma(beta_b) - (heads+tails)*special.digamma(beta_a+beta_b)
	ln_ro_nk = exp_pi + exp_theta
	ro_nk = np.exp(ln_ro_nk)
	return ro_nk

def calc_r(ro):
	sum_ro = np.sum(ro, axis=0)
	print(sum_ro)
	return ro/sum_ro

def update_dirichlet_param(dirichlet_param, r_nk):
	param_new = dirichlet_param + np.sum(r_nk,axis=0)	
	return param_new

def update_beta_param(bet_param, r, outcomes):
	pass

print(dirichlet_param)
print(np.shape(dirichlet_param))
print(r_nk)
print(ro_nk)

print(calc_r(ro_nk))
print(update_dirichlet_param(dirichlet_param, r_nk))
