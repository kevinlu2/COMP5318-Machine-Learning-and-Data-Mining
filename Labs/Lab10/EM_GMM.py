"""
EM algorithm for GMM
COMP5318 - Tutorial 10
"""

import numpy as np
import os.path
import matplotlib.pyplot as pl
from scipy.stats import multivariate_normal as mvn

#This is a function to generate data.
def gen_data(means, covs, weights, n_datapoints):
    n_datapoints = np.hstack((0, n_datapoints))
    data = np.empty((n_datapoints.sum(), 2))
    for i in range(len(means)):
        data[np.sum(n_datapoints[:(i+1)]):np.sum(n_datapoints[:(i+2)]), :] \
            = np.clip(weights[i]*np.random.multivariate_normal(means[i], covs[i], n_datapoints[i+1]), -10, 10)
    return data


#This is a function to plot
def my_plot(pi_k, mu_k, Sigma_k, log_likelihood_new_array, max_iter, X):
    n = 500
    xx, yy = np.meshgrid(np.linspace(-8, 8, n), np.linspace(-8, 8, n))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T

    pl.figure(figsize=(15, 8))
    ax = pl.subplot(121)
    for pi, mu, sigma in zip(pi_k, mu_k, Sigma_k):
        z = pi*mvn(mu, sigma).pdf(xy)
        pl.contour(xx, yy, z.reshape((n, n)), 10, linewidths=2)
    pl.colorbar()
    pl.scatter(X[:,0], X[:,1], c='', edgecolors='k')
    ax.axes.set_aspect('equal')
    pl.tight_layout()

    if len(log_likelihood_new_array) > 0:
        ax = pl.subplot(122)
        pl.scatter(np.arange(len(log_likelihood_new_array)), log_likelihood_new_array)
        pl.plot(np.arange(len(log_likelihood_new_array)), log_likelihood_new_array)
        ax.set_xlim([0, max_iter])
        ax.set_ylim([log_likelihood_new_array[0], 0])
        pl.tight_layout()
        pl.xlabel('Iteration')
        pl.ylabel('Log-likelihood')
    if not os.path.exists('EM_outs'):
        os.mkdir('EM_outs')
    pl.savefig('EM_outs/{}'.format(len(log_likelihood_new_array)))
    print('plotting {}'.format(len(log_likelihood_new_array)))

#gmm model - lab 8
def gmm(xs, tol=0.0001, max_iter=20):

    log_likelihood_new_array = []

    # Step 1 - Initial guesses for parameters
    pi_k = np.random.random(2)
    pi_k /= pi_k.sum()
    mu_k = np.random.random((2,2))*10
    Sigma_k = np.array([np.eye(2)] * 2)

    n, p = xs.shape
    k = len(pi_k)

    log_likelihood_old = 0
    for i in range(max_iter):
        #Let's plot!
        my_plot(pi_k, mu_k, Sigma_k, log_likelihood_new_array, max_iter, xs)

        # Step 2 - E-step
        ws = np.zeros((k, n))
        for j in range(len(mu_k)):
            for i in range(n):
                ws[j, i] = pi_k[j] * mvn(mu_k[j], Sigma_k[j], allow_singular=True).pdf(xs[i])
        ws /= ws.sum(0)

        # Step 3 - M-step        mu_k = np.zeros((k, p))
        for j in range(k):
            for i in range(n):
                mu_k[j] += ws[j, i] * xs[i]
            mu_k[j] /= ws[j, :].sum()

        Sigma_k = np.zeros((k, p, p))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(xs[i]- mu_k[j], (2,1))
                Sigma_k[j] += ws[j, i] * np.dot(ys, ys.T)
            Sigma_k[j] /= ws[j,:].sum()

        pi_k = np.zeros(k)
        for j in range(len(mu_k)):
            for i in range(n):
                pi_k[j] += ws[j, i]
        pi_k /= n

        # Step 4 - update complete log likelihoood
        log_likelihood_new = 0.0
        for i in range(n):
            s = 0
            for j in range(k):
                s += pi_k[j] * mvn(mu_k[j], Sigma_k[j], allow_singular=True).pdf(xs[i])
            log_likelihood_new += np.log(s)

        if np.abs(log_likelihood_new - log_likelihood_old) < tol:
            #print('breaking...')
            #break
            pass
        log_likelihood_old = log_likelihood_new

        log_likelihood_new_array.append(log_likelihood_new) #used for plotting purposes


    return log_likelihood_new, pi_k, mu_k, Sigma_k


def main():
    means = np.array([[3, -1], [-1, 0]])
    covs = [np.array([[4,  2], [2, 2]]),
            np.array([[1, -2], [-2, 3]])]

    mixing_coef = np.array([1, 1])

    n_datapoints = np.array([100, 200])

    X = gen_data(means, covs, mixing_coef, n_datapoints)

    log_likelihood, pi_k, mu_k, Sigma_k = gmm(X)
    print('pi={}, \n mu={}, \n Sigma={}'.format(pi_k, mu_k, Sigma_k))


if __name__ == "__main__":
    main()

