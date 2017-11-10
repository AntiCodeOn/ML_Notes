import CoinTossVI as vi
import unittest
import numpy as np
import numpy.testing as npt
import scipy.stats
import sys
from scipy import stats, special

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

dirichlet_param = np.array([[1.001, 1.001]], dtype=np.float64)
r_nk_ones = 1*np.ones((5,2))
beta_k1 = np.array([2, 2], dtype=np.float64)
beta_k2 = np.array([5, 3], dtype=np.float64)
beta_param = np.column_stack((np.transpose(beta_k1), np.transpose(beta_k2)))
dirichlet_old = dirichlet_param
beta_new = beta_param

def test_update_dirichlet_param():
        dirichlet_new = vi.update_dirichlet_param(dirichlet_old, r_nk_ones)
        expected = np.array([[6.001, 6.001]], dtype=np.float64)
        print(dirichlet_new)
        npt.assert_allclose(dirichlet_new, expected)
        npt.assert_equal(dirichlet_new.shape, expected.shape)

#digamma(1) = -0.5772
#digamma(2) = 0.4228
#digamma(3) = 0.9228
def test_psi_diff():
   diff_psi = vi.psi_diff(np.array([[1, 2]]))
   expected = np.array([[-1.5, -0.5]])
   npt.assert_allclose(diff_psi, expected)
   npt.assert_equal(np.shape(diff_psi), expected.shape)

def test_psi_diff2():
   diff_psi = vi.psi_diff(np.array([[2., 1.]]))
   expected = np.array([[-0.5, -1.5]])
   npt.assert_allclose(diff_psi, expected)
   npt.assert_equal(np.shape(diff_psi), expected.shape)

def test_sum_psi_diff():
   sum_diff_psi = vi.sum_psi_diff(np.array([[1., 2.], [2., 1.]]))
   expected = np.array([[-0.5], [-0.5]])
   npt.assert_allclose(sum_diff_psi, expected)
   npt.assert_equal(sum_diff_psi.shape, expected.shape)

if __name__ == '__main__':
   #unittest.main()
   npt.run_module_suite(argv=sys.argv)
