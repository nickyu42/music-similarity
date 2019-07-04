"""
Author: Nick Yu
Date created: 1/3/2019

Shounen/ Shoujo classifier
"""
import numpy as np
from sklearn import mixture


def fit_gmm(mfcc, n_comp=20):
    """
    Fits gaussian mixture on mfccs
    :param mfcc: np.ndarray of dtype=complex, each row should be a sample
    :param n_comp: number of mixture components to fit on
    :return: flattened feature vector of weights, means and full covariance matrix
    """
    mfcc = np.nan_to_num(mfcc.real)

    gmix = mixture.GaussianMixture(n_components=n_comp, covariance_type='full')
    gmix.fit(mfcc)

    # the size of each should be
    # weights       = (n_comp,)
    # means         = (n_comp, n_feat)
    # covariances   = (n_comp, n_feat, n_feat)
    # assuming default values, n_comp = 20 and n_features = 12 mfcc features
    features = gmix.weights_, gmix.means_, gmix.covariances_
    return np.concatenate([f.flatten() for f in features])


def init_gmm(gmm_params, n_comp=20, n_feat=12):
    """
    Generate random samples from generated gmm based on gmm parameters
    :param gmm_params: vector of concatenated weights, means and covariances
    :param n_comp: number of mixture components used
    :param n_feat: feature vector length
    :return: array of n_features sized samples
    """
    gmix = mixture.GaussianMixture(n_components=n_comp, covariance_type='full')

    # hacky method to bypass gmix._check_is_fitted()
    gmix.fit(np.random.rand(n_comp, 12))

    gmix.weights_ = gmm_params[:n_comp]
    gmix.means_ = np.reshape(gmm_params[n_comp:n_comp + n_comp*n_feat], (n_comp, n_feat))
    gmix.covariances_ = np.reshape(gmm_params[n_comp + n_comp*n_feat:], (n_comp, n_feat, n_feat))

    return gmix


def gmm_js(gmm_p: mixture.GaussianMixture, gmm_q: mixture.GaussianMixture, sample_count=500):
    """
    Calculates Jensen-Shannon divergence of two gmm's
    :param gmm_p: mixture.GaussianMixture
    :param gmm_q: mixture.GaussianMixture
    :param sample_count: number of monte carlo samples to use
    :return: Jensen-Shannon divergence
    """
    X = gmm_p.sample(sample_count)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(sample_count)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    # black magic?
    return (log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2


def gmm_kl(gmm_p, gmm_q, sample_count=500):
    X = gmm_p.sample(sample_count)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    return log_p_X.mean() - log_q_X.mean()


def gmm_symmetric_kl(gmm_p, gmm_q, sample_count=500):
    """
    Calculates symmetric Kullback-Leibler divergence
    defined as KL_S = KL(p || q) + KL(q || p)
    :param gmm_p: mixture.GaussianMixture
    :param gmm_q: mixture.GaussianMixture
    :param sample_count: number of monte carlo samples to use
    :return: Kullback-Leibler divergence
    """
    return gmm_kl(gmm_p, gmm_q, sample_count=sample_count) + \
           gmm_kl(gmm_q, gmm_p, sample_count=sample_count)
