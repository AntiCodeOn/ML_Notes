import numpy as np
import scipy.stats as scs
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(733)

matplotlib.rcParams.update({'font.size': 14})
plt.style.use(['seaborn-pastel', 'seaborn-white'])

def plot_beta(beta_param):
   z = np.linspace(0, 1, 250)

   beta1 = np.transpose(beta_param[:,0])
   beta2 = np.transpose(beta_param[:,1])

   p_z1 = scs.beta(beta1[0], beta1[1]).pdf(z)
   p_z2 = scs.beta(beta2[0], beta2[1]).pdf(z)

   plt.figure(figsize=(12, 7))
   plt.plot(z, p_z1, linewidth=3.)
   plt.plot(z, p_z2, linewidth=2.)
   plt.axvline(0.45, color='red')
   plt.axvline(0.80, color='red')
   plt.axvline(0.79, color='blue')
   plt.axvline(0.52, color='orange')
   plt.ylabel('p(Theta)')
   plt.show()

def plot_elbo(elbo):
   z = np.linspace(1, len(elbo), len(elbo))

   plt.figure(figsize=(12, 7))
   plt.plot(z, elbo, linewidth=3.)
   plt.axhline(elbo[-1], color='red')
   plt.ylabel('ELBO')
   plt.show()
