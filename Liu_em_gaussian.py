#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_em_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as LastName_em_gaussian.py (replacing LastName with your last name)!')
    exit(1)

DATA_PATH = "/u/cs246/data/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    if args.cluster_num:
        lambdas = np.zeros(args.cluster_num)
        mus = np.zeros((args.cluster_num,2))
        if not args.tied:
            sigmas = np.zeros((args.cluster_num,2,2))
        else:
            sigmas = np.zeros((2,2))
        #TODO: randomly initialize clusters (lambdas, mus, and sigmas)
        train_xs, dev_xs = parse_data(args)

        chosen = np.random.choice(len(train_xs), args.cluster_num, replace=False)
        lambdas = np.array([1.0 / args.cluster_num] * args.cluster_num)
        mus = np.array([train_xs[x] for x in chosen])
        if not args.tied:
            # sigmas = np.zeros((args.cluster_num,2,2))
            sigmas = np.array([np.cov(train_xs, rowvar = 0)] * args.cluster_num)
        else:
            # sigmas = np.zeros((2,2))
            sigmas = np.array(np.cov(train_xs, rowvar = 0))
        # raise NotImplementedError #remove when random initialization is implemented
    else:
        lambdas = []
        mus = []
        sigmas = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #lambda mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float,line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2])
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]])
        lambdas = np.asarray(lambdas)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)

    #TODO: do whatever you want to pack the lambdas, mus, and sigmas into the model variable (just a tuple, or a class, etc.)
    #NOTE: if args.tied was provided, sigmas will have a different shape
    model = [lambdas, mus, sigmas]
    # raise NotImplementedError #remove when model initialization is implemented
    return model


def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #NOTE: you can use multivariate_normal like this:
    #probability_of_xn_given_mu_and_sigma = multivariate_normal(mean=mu, cov=sigma).pdf(xn)
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
    # raise NotImplementedError #remove when model training is implemented

    ### log_likelihoods of every iterations
    log_likelihoods = []

    # for dev data to choose best model
    best_model = model
    best_ll = float("-inf")
    best_iter = 0


    lambdas = model[0]
    mus = model[1]
    sigmas = model[2]
    N = len(train_xs)   # number of training samples
    
    current_iter = 0
    train_xs = np.matrix(train_xs)

    # P(Z|X)
    P_Z_given_X = np.zeros([N, args.cluster_num])
    # print("mus: ", mus)
    # print("sigmas: ", sigmas)
    # print("lambdas: ", lambdas)

    while current_iter < args.iterations:
        
        """ E step """

        for k in range(args.cluster_num):
            if not args.tied:
                P_Z_given_X[:, k] = lambdas[k] * N_X_given_muk_sigmak(train_xs, mus[k], sigmas[k])
            else:
                P_Z_given_X[:, k] = lambdas[k] * N_X_given_muk_sigmak(train_xs, mus[k], sigmas)


        
        ## Normalize to make the matrix row stochastic
        P_Z_given_X = (P_Z_given_X.T / np.sum(P_Z_given_X, axis = 1)).T
        ## The number of datapoints belonging to each cluster           
        Num_ks = np.sum(P_Z_given_X, axis = 0)

        """ M step: calculate the new mus and sigmas for each gaussian by applying above P_Z_given_X """
        for k in range(args.cluster_num):
            # lambdas update
            lambdas[k] = 1.0 / N * Num_ks[k]

            # mus update
            total = np.matrix([0.0, 0.0])
            for i in range(N):
                # print("********P_Z_given_X[k][i]: ", P_Z_given_X[k][i])
                temp2 = P_Z_given_X[i][k] * train_xs[i]

                total += temp2
            mus[k] = total / Num_ks[k]
            x_minus_mus = np.matrix(train_xs - mus[k])


            # sigmas updata
            if not args.tied:
                sigmas[k] = np.array(1.0 / Num_ks[k] * np.dot(np.multiply(x_minus_mus.T, P_Z_given_X[:, k]), x_minus_mus))
            else:
                sigmas = np.array(1.0 / Num_ks[k] * np.dot(np.multiply(x_minus_mus.T, P_Z_given_X[:, k]), x_minus_mus))

        ## likelihood computation for plotting
        current_model = [lambdas, mus, sigmas]
        # current_log_likelihood = np.sum(np.log(np.sum(P_Z_given_X, axis = 1)))
        current_log_likelihood = average_log_likelihood(current_model, train_xs, args)
        log_likelihoods.append(current_log_likelihood)

        current_iter += 1
        if not args.nodev:
            ll_dev = average_log_likelihood(current_model, dev_xs, args)
            print("iter %s dev log_likelihood: %s" % (str(current_iter), str(ll_dev)))
            if ll_dev > best_ll:
                best_ll = ll_dev
                best_model = current_model
                best_iter = current_iter
        print("iter %s train log_likelihood: %s" % (str(current_iter), str(current_log_likelihood)))

        
        # # check for convergence
        # if len(log_likelihoods) < 2:
        #     continue
        # if np.abs(current_log_likelihood - log_likelihoods[-2]) < 0.00001:
        #     break

    model = [lambdas, mus, sigmas]

    plot_log_likelihood(log_likelihoods)
    # demo_2d(train_xs, mus, sigmas, log_likelihoods)

    if not args.nodev:
        print("best iterations:", str(best_iter))
        return best_model
    return model


def average_log_likelihood(model, data, args):
    from math import log
    from scipy.stats import multivariate_normal
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    ll = 0.0
    lambdas = model[0]
    mus = model[1]
    sigmas = model[2]
    N = len(data)
    P_Z_given_X = np.zeros([N, args.cluster_num])


    for k in range(args.cluster_num):
        if not args.tied:
            P_Z_given_X[:, k] = lambdas[k] * N_X_given_muk_sigmak(data, mus[k], sigmas[k])
        else:
            P_Z_given_X[:, k] = lambdas[k] * N_X_given_muk_sigmak(data, mus[k], sigmas)

    ## likelihood computation for plotting
    ll = 1.0 / N * np.sum(np.log(np.sum(P_Z_given_X, axis = 1)))

    # raise NotImplementedError #remove when average log likelihood calculation is implemented
    return ll

def extract_parameters(model):
    #TODO: extract lambdas, mus, and sigmas from the model and return them (same type and shape as in init_model)
    lambdas = model[0]
    mus = model[1]
    sigmas = model[2]
    # raise NotImplementedError #remove when parameter extraction is implemented
    return lambdas, mus, sigmas


def N_X_given_muk_sigmak(X, mu_k, sigma_k):
    from scipy.stats import multivariate_normal
    # Calculate the probability for X in normal distribution k
    # mu_k stands for mus[k]
    # sigma_k stands for sigmas[k]

    probability_matrix = []
    for xn in X:
        probability = multivariate_normal(mean=mu_k, cov=sigma_k).pdf(xn)
        probability_matrix.append(probability)
    probability_matrix = np.matrix(probability_matrix)

    return probability_matrix

def plot_log_likelihood(log_likelihoods):
    import pylab as plt
    # print("log_likelihoods: ", log_likelihoods)
    plt.plot(log_likelihoods)
    plt.title('Log Likelihood vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    plt.show()

def demo_2d(data, mus, sigmas, log_likelihoods):
    import pylab as plt    
    from matplotlib.patches import Ellipse
    
    def plot_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]
    
        if ax is None:
            ax = plt.gca()
    
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    
        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(abs(vals))
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    
        ax.add_artist(ellip)
        return ellip    
    
    def show(X, mu, cov):

        plt.cla()
        K = len(mu) # number of clusters
        colors = ['b', 'k', 'g', 'c', 'm', 'y', 'r']
        plt.plot(X.T[0], X.T[1], 'm*')
        for k in range(K):
          plot_ellipse(mu[k], cov[k],  alpha=0.6, color = colors[k % len(colors)])  

    
    fig = plt.figure(figsize = (13, 6))
    fig.add_subplot(121)
    show(data, mus, sigmas)
    fig.add_subplot(122)
    plt.plot(np.array(log_likelihoods))
    plt.title('Log Likelihood vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    plt.show()
       


def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points.')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    ll_train = average_log_likelihood(model, train_xs, args)

    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()