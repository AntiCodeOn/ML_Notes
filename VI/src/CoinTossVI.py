import numpy as np
import operator
import matplotlib.pyplot as plt
from scipy import stats, special

import pyprint
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
heads = np.sum(bin_events, axis = 1)
tails = len(events[0]) - heads
experiments = np.column_stack((heads, tails))

iteration_stop = True

def convergence_policy(elbo_old, elbo_new, i, iterations):
   #TODO: convert to closure
   if (iteration_stop):
      return True
   else:
      return (abs(elbo_new - elbo_old) > 0.001)


def vi_algorithm(params_list = [], snapshot_list = []):
   elbo_log = []
   beta_log = []
   snapshot = []
   for param in params_list: 
      dirichlet_param, beta_param = param[0], param[1]
      iterations = param[2]
      elbo_history = []
      elbo_old = 0
      for i in range(iterations):
         r_nk = via.calc_r(experiments, dirichlet_param, beta_param)

         dirichlet_new = via.update_dirichlet_param(dirichlet_param, r_nk)
         beta_new = via.update_beta_param2(beta_param, r_nk, experiments)
         elbo_new = via.ELBO(r_nk, dirichlet_param, beta_param, dirichlet_new, beta_new, experiments)

         dirichlet_param = dirichlet_new
         beta_param = beta_new
         if i in snapshot_list:
            print(i)
            snapshot.append([r_nk, beta_new])

         if convergence_policy(elbo_old, elbo_new, i, iterations):
            elbo_old = elbo_new
            elbo_history.append(elbo_new)
         else:
            break
      elbo_log.append(elbo_history)
      beta_log.append([param[1], beta_param])

      #pyprint.plot_beta(beta_param)
   return elbo_log, beta_log, snapshot

def create_random_params(iterations = 500):
      """
         Create random parameters for dirichlet and Bernoulli
      """
      dirichlet_param = np.random.rand(1,2)
      dirichlet_param = np.array([[0.4, 0.6]])
      beta_param = np.random.randint(1, 10, size=(2, 2))
      b1 = 3 + 2*(np.random.rand() - 0.5)
      a2 = 3 + 2*(np.random.rand() + 0.5)
      beta_k1 = np.array([3, b1], dtype=np.float64)
      beta_k2 = np.array([a2, 3], dtype=np.float64)
      beta_param = np.column_stack((np.transpose(beta_k1), np.transpose(beta_k2)))
      return [dirichlet_param, beta_param, iterations]

def experiment1():
   """
      ELBO Convergence plot
   """
   iteration_stop = True #stop after predefined number of iterations
   iterations = 500
   params_list = []
   for i in range(100):
      params = create_random_params(iterations)
      params_list.append(params)
   elbo_log, _ , _ = vi_algorithm(params_list)
   pyprint.plot_elbo(elbo_log)


def experiment2():
   iteration_stop = False #converge on elbo difference
   iterations = 1000
   params_list = []
   for i in range(4):
      params = create_random_params(iterations)
      params_list.append(params)
   _, beta_log, _ = vi_algorithm(params_list)
   for beta in beta_log:
      pyprint.plot_beta(beta[0], beta[1])

def experiment3():
   iteration_stop = True
   iterations = 20
   param_list = []
   snapshot_list = [0, 10, 19]
   params_list = []
   params_list.append(create_random_params(iterations))
   _, beta_log, snapshot = vi_algorithm(params_list, snapshot_list)
   betas = []
   snapshots = []
   for i in range(len(snapshot_list)):
      #pyprint.plot_single_beta(snapshot[i][1])
      print(snapshot[i][0])
      snapshots.append([snapshot_list[i], snapshot[i][1]])
   pyprint.subplot_beta(snapshots)


dirichlet_param1 = np.array([[0.501, 0.501]], dtype=np.float64)
beta_k1 = np.array([2, 1], dtype=np.float64)
beta_k2 = np.array([3, 5], dtype=np.float64)
beta_param1 = np.column_stack((np.transpose(beta_k1), np.transpose(beta_k2)))
dirichlet_param2 = np.array([[0.401, 0.801]], dtype=np.float64)
beta_k1 = np.array([1, 2], dtype=np.float64)
beta_k2 = np.array([8, 3], dtype=np.float64)
beta_param2 = np.column_stack((np.transpose(beta_k1), np.transpose(beta_k2)))

iterations = 500
params1 = [dirichlet_param1, beta_param1, iterations]
params2 = [dirichlet_param2, beta_param2, iterations]

if __name__ == '__main__':
   #experiment1()
   #experiment2()
   experiment3()

