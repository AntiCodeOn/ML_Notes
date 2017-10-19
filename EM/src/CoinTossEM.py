import numpy as np
import operator
from scipy import stats
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
experiments = list(zip(heads, tails))
print(coins)
print(heads)

#random theta guess
rTheta = [0.6, 0.5]

#t, T, s = sp.symbols('theta, T, s')
#likelihood = (t**s)*(1-t)**(T-s)
#_likelihood = sp.lambdify((t,T,s), likelihood, modules='numpy')
diff = 0.00001

for i in range(10000):
   E = np.array([expect_zk(rTheta, experiment) for experiment in experiments])
   rThetaPrev = rTheta
   rTheta = max_likelihood(E, bin_events)
   diff0 = rThetaPrev[0]-rTheta[0]
   diff1 = rThetaPrev[1]-rTheta[1]
   if((abs(diff0) < diff) and (abs(diff1) < diff)):
      print('iterations until convergence: ', i+1)
      break
	
print(rTheta)

