#[[int(letter=='H') for letter in word] for word in words]

import numpy as np
import operator
from scipy import stats
import sympy as sp

#true (unknown theta values)
Theta = [0.8, 0.45]
#true (unknown coin selections)
coins  = ['B', 'A', 'A', 'B', 'A']

events = (['HTTTHHTHTH',
          'HHHHTHHHHH',
          'HTHHHHHTHH',
          'HTHTTTHHTT',
          'THHHTHHHTH'])
bin_events = [[ for event in events]
Total = len(events[0])

heads = [event.count('H') for event in events]
print(coins)
print(heads)

#random theta guess
rTheta = [0.6, 0.7]

t, T, s = sp.symbols('theta, T, s')
likelihood = (t**s)*(1-t)**(T-s)
_likelihood = sp.lambdify((t,T,s), likelihood, modules='numpy')
Ez = [0, 0]
i = 0
for j in range(events):
   Ez[i]=_likelihood(rTheta[i], Total, heads[j])
   Ez[i+1]=_likelihood(rTheta[i+1], Total, heads[j])

