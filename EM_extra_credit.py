import numpy as np
from scipy.stats import multivariate_normal as gaussian
from sklearn.mixture import GaussianMixture

X = np.loadtxt('X_new.txt')
no_of_mixtures = 3
gmm = GaussianMixture(n_components=no_of_mixtures, covariance_type='spherical', init_params = 'random')
gmm.fit(X)   
sigma = gmm.covariances_

mu_prev = np.random.randn(no_of_mixtures, X.shape[1])
weights_old = np.random.randn(no_of_mixtures)

prob_y_given_x = 0.1*np.zeros((no_of_mixtures, X.shape[0]))
error_combined = 1.

while error_combined > 0.001:
    for i in range(X.shape[0]):
        for j in range(no_of_mixtures):
            prob_y_given_x[j,i] = weights_old[j] * gaussian.pdf(X[i], mu_prev[j], np.identity(X.shape[1]))
        normalization_term = sum(prob_y_given_x[:,i])
        for j in range(no_of_mixtures):
            prob_y_given_x[j,i] = prob_y_given_x[j,i]/normalization_term
    
    mu_curr = np.zeros((no_of_mixtures,X.shape[1]))
    weights_new = np.zeros(no_of_mixtures)
    for j in range(no_of_mixtures):
        for i in range(X.shape[0]):
            mu_curr[j] = mu_curr[j] + prob_y_given_x[j,i]*X[i]
        mu_curr[j] = mu_curr[j]/sum(prob_y_given_x[j,:])
        
        weights_new[j] = sum(prob_y_given_x[j,:])/X.shape[0]

    error_combined = np.linalg.norm(mu_curr - mu_prev) + np.linalg.norm(weights_new - weights_old)
    mu_prev = mu_curr
    weights_old = weights_new

for idx in range(3):
    print "mean_{}".format(idx), gmm.means_[idx,:]
    print "weight_{}".format(idx), gmm.weights_[idx]

print "error_combined", gmm.means_-mu_curr
