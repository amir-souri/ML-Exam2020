import util as u
import dset as d
import numpy as np
from matplotlib import pyplot as plt

path = ".../IMM-Frontal Face DB SMALL"

shapes, images = d.face_shape_data(path=path)
mean = np.mean(shapes, axis=0)

def pca(X):
    μ = np.mean(X, axis=0)
    centered_X = X - μ
    cov = np.cov(centered_X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvectors = eigenvectors.T
    idxs = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[idxs]
    components = eigenvectors
    return eigenvalues, components

def features_to_components(features, W, mu):
    """from feature space to principal component space"""
    centered_X = features - mu
    return W @ centered_X.T

def components_to_features(vector, W, mu):
    """from principal component space to feature space"""
    X = W.T @ vector
    return X.T + mu

def rmse(X, W, mu):
    X_prime = features_to_components(features=X, W=W, mu=mu)
    XX = components_to_features(X_prime, W, mu)
    return np.sqrt(np.square(np.subtract(X, XX)).mean())

def proportional_variance(eig_vals):
    tot = sum(eig_vals)
    var_exp = [(i / tot) for i in eig_vals]
    return var_exp

def cumulative_proportional_variance(eig_vals):
    tot = sum(eig_vals)
    var_exp = [(i / tot) for i in eig_vals]
    cum_var_exp = np.cumsum(var_exp)
    return cum_var_exp

eig_vals, W_all = pca(X=shapes)
rmse_all = []
for i in range(1, shapes.shape[1] + 1):
    rmse_all.append(rmse(shapes, W_all[:i], mean))

fig, ax = plt.subplots(1, 2, figsize=(12, 12))
x = np.arange(1, shapes.shape[1] + 1)
ax[0].plot(x, rmse_all)
ax[0].set_xlabel('number of components', fontdict={'size': 21})
ax[0].set_ylabel('Rconstruction Error', fontdict={'size': 21})
ax[0].tick_params(axis='both', labelsize=14)
ax[1].plot(x, np.real(cumulative_proportional_variance(eig_vals)), 'r')
ax[1].plot(x, np.real(proportional_variance(eig_vals)), 'b')
ax[1].set_xlabel('number of components', fontdict={'size': 21})
ax[1].set_ylabel('Variance (fraction of total)', fontdict={'size': 21})
ax[1].axis(xmin=0,xmax=80)
ax[1].legend(labels=('Cumulative variance', 'Variance proportion'), prop={'size': 12})
ax[1].tick_params(axis='both', labelsize=14)

X_prime = features_to_components(features=shapes, W=W_all, mu=mean)
std = np.std(X_prime, axis=1)
sliderplot = u.SliderPlot(principal_components=W_all, std=std, mu=mean,
                          inverse_transform=components_to_features)
plt.show()
