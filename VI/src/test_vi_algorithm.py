import unittest
import vi_algorithm as via
import numpy as np
import numpy.testing as npt
import scipy.stats
import sys
from scipy import stats, special

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

dirichlet_param = np.array([[1.001, 1.001]], dtype=np.float64)
r_nk_ones = 1*np.ones((5,2))
beta_k1 = np.array([2, 2], dtype=np.float64)
beta_k2 = np.array([5, 3], dtype=np.float64)
beta_param = np.column_stack((np.transpose(beta_k1), np.transpose(beta_k2)))
dirichlet_old = dirichlet_param
beta_new = beta_param

#digamma(1) = -0.5772
#digamma(2) = 0.4228
#digamma(3) = 0.9228


if __name__ == '__main__':
   #unittest.main()
   npt.run_module_suite(argv=sys.argv)
