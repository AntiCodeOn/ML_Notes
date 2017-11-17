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

#digamma(1) = -0.5772
#digamma(2) = 0.4228
#digamma(3) = 0.9228
def test_calc_r():
   r_nk = calc_r(experiments, dirichlet_param, beta_param)

def test_update_dirichlet_param():
   dirichlet_old = np.array([[0.5, 1.5]])
   r_nk = np.arange(6).reshape(3, 2)
   dirichlet_new = via.update_dirichlet_param(dirichlet_old, r_nk)
   expected = np.array([[6.5, 10.5]])
   npt.assert_equal(dirichlet_new, expected)

def test_update_beta_param():
   beta_old = beta_param
   r_nk = np.arange(6).reshape(3, 2)
   outcomes = np.array([[3, 0], [2, 1], [1, 2]])
   expected = np.array([[10, 19], [12, 16]])
   beta_new = via.update_beta_param2(beta_old, r_nk, outcomes)
   npt.assert_equal(beta_new, expected)

if __name__ == '__main__':
   #unittest.main()
   npt.run_module_suite(argv=sys.argv)
