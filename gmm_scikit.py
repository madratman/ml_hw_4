import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

x_input = np.loadtxt('X_new.txt')
color_list = ['blue','green','red']

gmm = GaussianMixture(n_components=3, covariance_type='spherical', init_params = 'random')
gmm.fit(x_input)
y_pred =  gmm.predict(x_input)

for idx in range(3):
	print "mean_{}".format(idx), gmm.means_[idx]
	print "sigma_{}".format(idx), gmm.covariances_[idx]
	print "weight_{}".format(idx), gmm.weights_[idx]

mean_x_axis = gmm.means_[:,0]
mean_indices = sorted(range(len(mean_x_axis)), key=lambda k: mean_x_axis[k])
color_list_sort = [color_list[idx] for idx in mean_indices]
# (x1,x2), (x3,x4), (x4,x5)
dims_list = [(1,2), (3,4), (4,5)]
plot_lims = 10
for (ctr,dims) in enumerate(dims_list):
	plt.figure()
	for idx in range(3):
		plt.scatter(x_input[y_pred==idx,dims[0]-1], x_input[y_pred==idx,dims[1]-1], .8, color=color_list_sort[idx])
		plt.xlim(-plot_lims, plot_lims)
		plt.ylim(-plot_lims, plot_lims)
		plt.xticks(range(-plot_lims, plot_lims, 1))
		plt.yticks(range(-plot_lims, plot_lims, 1))
		plt.title("dims x{} and x{}".format(dims[0], dims[1]))
	plt.savefig('plot_{}.png'.format(ctr))
plt.show()
