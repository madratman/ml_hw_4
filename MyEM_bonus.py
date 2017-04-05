import numpy as np
from scipy.stats import multivariate_normal as gaussian
from sklearn.mixture import GaussianMixture

X = np.loadtxt('X_new.txt')
no_of_mixtures = 3
PI =  (1./3, 1./3, 1./3)
gmm = GaussianMixture(n_components=no_of_mixtures, covariance_type='spherical', init_params = 'random')
gmm.fit(X)   
sigma = gmm.covariances_

mu_old = np.random.randn(no_of_mixtures, X.shape[1])
PI_old = np.random.randn(no_of_mixtures)

error = 1
Py_x = .1*np.zeros((no_of_mixtures, X.shape[0])) #probability of y given x

while error > 0.001:
    # expectation
    for i in range(X.shape[0]):
        for j in range(no_of_mixtures):
            Py_x[j,i] = PI_old[j] * gaussian.pdf(X[i], mu_old[j], np.identity(X.shape[1]))
        
        S = sum(Py_x[:,i])
        
        for j in range(no_of_mixtures):
            Py_x[j,i] = Py_x[j,i]/S
    
    # maximization
    mu_new = np.zeros((no_of_mixtures,X.shape[1]))
    PI_new = np.zeros(no_of_mixtures)
    for j in range(no_of_mixtures):
        for i in range(X.shape[0]):
            mu_new[j] = mu_new[j] + Py_x[j,i]*X[i]
        mu_new[j] = mu_new[j]/sum(Py_x[j,:])
        
        PI_new[j] = sum(Py_x[j,:])/X.shape[0]

    error = np.linalg.norm(mu_new - mu_old) + np.linalg.norm(PI_new - PI_old)
    mu_old = mu_new
    PI_old = PI_new

## print the means 
print('The means of the 3 Gaussians are:\n')
for i in range(3):
    print('mu_'+`i+1` + ':')
    print mu_new[i,:]

## print the wrights (pi's) 
print('The weights of the 3 Gaussians are:\n')
for i in range(3):
    print('pi_'+`i+1` + ':')
    print PI_new[i]

print ('error between Sklearn means and my means')
e = gmm.means_ - mu_new
print e