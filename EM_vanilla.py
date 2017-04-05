import numpy as np
from scipy.stats import multivariate_normal as gaussian
from sklearn.mixture import GaussianMixture

X = np.loadtxt('X_new.txt')
no_of_mixtures = 3
gmm = GaussianMixture(n_components=no_of_mixtures, covariance_type='spherical', init_params = 'random')
gmm.fit(X)
weights = gmm.weights_
sigma = gmm.covariances_
prob_y_given_x = 0.1*np.zeros((no_of_mixtures, X.shape[0]))
mu_prev = np.random.randn(no_of_mixtures, X.shape[1])
error = 1.

while error > 0.001:
	for i in range(X.shape[0]):
		for j in range(no_of_mixtures):
			prob_y_given_x[j,i] = weights[j] * gaussian.pdf(X[i], mu_prev[j], np.identity(X.shape[1]))		
		normalization_term = sum(prob_y_given_x[:,i])
		for j in range(no_of_mixtures):
			prob_y_given_x[j,i] = prob_y_given_x[j,i]/normalization_term

	mu_curr = np.zeros((no_of_mixtures,X.shape[1]))
	for j in range(no_of_mixtures):
		for i in range(X.shape[0]):
			mu_curr[j] = mu_curr[j] + prob_y_given_x[j,i]*X[i]
		mu_curr[j] = mu_curr[j]/sum(prob_y_given_x[j,:])

	error = np.linalg.norm(mu_curr - mu_prev)
	mu_prev = mu_curr

for idx in range(3):
	print "mean_old_{}".format(idx), gmm.means_[idx,:]
	print "mean_new_{}".format(idx), mu_curr[idx,:]

print "error", gmm.means_-mu_curr

