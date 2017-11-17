import unittest
import vi_statistic as vis
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

def test_logC():
   for a in range(1, 20, 2):
      npt.assert_equal(vis.logC(np.array([[a]])), 0.0)

def test_logC2():
   npt.assert_allclose(vis.logC(np.array([[1., 2., 3.]])), 4.09434456222)

def test_psi_diff():
   diff_psi = vis.psi_diff(np.array([[1., 2.]]))
   expected = np.array([[-1.5, -0.5]])
   npt.assert_allclose(diff_psi, expected)
   npt.assert_equal(np.shape(diff_psi), expected.shape)

def test_psi_diff2():
   diff_psi = vis.psi_diff(np.array([[2., 1.]]))
   expected = np.array([[-0.5, -1.5]])
   npt.assert_allclose(diff_psi, expected)
   npt.assert_equal(np.shape(diff_psi), expected.shape)

def test_psi_diff3():
   diff_psi = vis.psi_diff(np.array([[1., 2.],[2., 1.]]))
   expected = np.array([[-1.5, -0.5],[-0.5, -1.5]])
   npt.assert_allclose(diff_psi, expected)
   npt.assert_equal(np.shape(diff_psi), expected.shape)

def test_exp_pi():
   param = np.array([1., 2])
   expected = -1*stats.dirichlet.entropy(param)
   p_pi = vis.exp_p_pi(param.reshape(1,2))
   print("expected ", expected)
   print("p_pi ", p_pi)
   npt.assert_equal(p_pi, expected)


def test_sum_psi_diff():
   sum_diff_psi = vis.sum_psi_diff(np.array([[1., 2.], [2., 1.]]))
   expected = np.array([[-0.5], [-0.5]])
   npt.assert_allclose(sum_diff_psi, expected)
   npt.assert_equal(sum_diff_psi.shape, expected.shape)

def test_exp_p_theta():
   expected1 = -1*stats.beta.entropy(1., 2.)
   expected2 = -1*stats.beta.entropy(2., 1.)
   print("expected1", expected1)
   print("expected2", expected2)
   p_theta = vis.exp_p_theta([[1., 2.],[2., 1.]])
   npt.assert_equal(p_theta, [expected1, expected2])

def test_exp_p_Z():
   r_nk = np.ones((5,2))
   p_Z = vis.exp_p_Z(r_nk, np.array([[1., 2.]]))
   expected = -10.
   npt.assert_allclose(p_Z, expected)

def test_exp_q_Z():
   r_nk = np.exp(1)*np.ones((5,2))
   q_Z = vis.exp_q_Z(r_nk)
   expected = 10*np.exp(1)
   npt.assert_allclose(q_Z, expected)

if __name__ == '__main__':
   #unittest.main()
   npt.run_module_suite(argv=sys.argv)
