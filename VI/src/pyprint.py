import numpy as np
import scipy.stats as scs
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(733)

matplotlib.rcParams.update({'font.size': 14})
plt.style.use(['seaborn-pastel', 'seaborn-white'])

def plot_beta(beta_old, beta_new):
   z = np.linspace(0, 1, 250)

   beta_old1 = np.transpose(beta_old[:,0])
   beta_old2 = np.transpose(beta_old[:,1])

   p_z1 = scs.beta(beta_old1[0], beta_old1[1]).pdf(z)
   p_z2 = scs.beta(beta_old2[0], beta_old2[1]).pdf(z)

   plt.figure(figsize=(12, 7))
   plt.plot(z, p_z1, linewidth=1.)
   plt.plot(z, p_z2, linewidth=1.)

   beta_new1 = np.transpose(beta_new[:,0])
   beta_new2 = np.transpose(beta_new[:,1])

   p_z1 = scs.beta(beta_new1[0], beta_new1[1]).pdf(z)
   p_z2 = scs.beta(beta_new2[0], beta_new2[1]).pdf(z)
   plt.plot(z, p_z1, linewidth=1.)
   plt.plot(z, p_z2, linewidth=1.)
   #plt.axvline(0.45, color='red')
   #plt.axvline(0.80, color='red')
   #plt.axvline(0.79, color='blue')
   #plt.axvline(0.52, color='orange')
   plt.ylabel('p(Theta)')
   plt.show()

def plot_elbo(elbos):
   plt.figure(figsize=(12, 7))
   
   avg = 0
   for elbo in elbos:
      z = np.linspace(1, len(elbo), len(elbo))

      plt.plot(z, elbo, linewidth=1.)
      avg += elbo[-1]
   avg = avg /len(elbos)
   plt.axhline(avg, color='red')
   plt.xlabel('iterations')
   plt.ylabel('ELBO')
   plt.show()
