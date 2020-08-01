"""
Author: Nick Yu  
Date created: 1/3/2019  

Shounen/ Shoujo classifier
"""
from typing import Callable
import numpy as np
from sklearn import mixture

from .processing import convert_to_mfcc


def fit_gmm(mfcc, n_comp=20):
    """
    Fits gaussian mixture on mfccs
    :param mfcc: np.ndarray of dtype=complex, each row should be a sample
    :param n_comp: number of mixture components to fit on
    :return: flattened feature vector of weights, means and full covariance matrix
    """
    mfcc = np.nan_to_num(mfcc.real).T

    gmix = mixture.GaussianMixture(n_components=n_comp, covariance_type='full')
    gmix.fit(mfcc)

    # the size of each should be
    # weights       = (n_comp,)
    # means         = (n_comp, n_feat)
    # covariances   = (n_comp, n_feat, n_feat)
    # assuming default values, n_comp = 20 and n_features = 12 mfcc features
    features = gmix.weights_, gmix.means_, gmix.precisions_cholesky_
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
    gmix.fit(np.random.rand(n_comp, n_feat))

    gmix.weights_ = gmm_params[:n_comp]
    gmix.means_ = np.reshape(
        gmm_params[n_comp:n_comp + n_comp*n_feat], (n_comp, n_feat))
    gmix.precisions_cholesky_ = np.reshape(
        gmm_params[n_comp + n_comp*n_feat:], (n_comp, n_feat, n_feat))

    return gmix


def calculate_distance(mfcc1: np.ndarray, mfcc2: np.ndarray, distance_measure: Callable) -> float:
    """
    Calculates distance based on given measure
    :param mfcc1: mfcc frames of song 1
    :param mfcc2: mfcc frames of song 2
    :param distance_measure: measure to use
    :return: 'distance' of given songs
    """
    gmix1 = mixture.GaussianMixture(n_components=20, covariance_type='diag')
    gmix1.fit(np.nan_to_num(mfcc1.real).T)

    gmix2 = mixture.GaussianMixture(n_components=20, covariance_type='diag')
    gmix2.fit(np.nan_to_num(mfcc2.real).T)

    return distance_measure(gmix1, gmix2)


def gmm_js(gmm_p: mixture.GaussianMixture, gmm_q: mixture.GaussianMixture, sample_count=500):
    """
    Calculates Jensen-Shannon divergence of two gmm's
    :param gmm_p: mixture.GaussianMixture
    :param gmm_q: mixture.GaussianMixture
    :param sample_count: number of monte carlo samples to use
    :return: Jensen-Shannon divergence
    """
    x = gmm_p.sample(sample_count)[0]
    log_p_x = gmm_p.score_samples(x)
    log_q_x = gmm_q.score_samples(x)
    log_mix_x = np.logaddexp(log_p_x, log_q_x)

    y = gmm_q.sample(sample_count)[0]
    log_p_y = gmm_p.score_samples(y)
    log_q_y = gmm_q.score_samples(y)
    log_mix_y = np.logaddexp(log_p_y, log_q_y)

    # black magic?
    return (log_p_x.mean() - (log_mix_x.mean() - np.log(2))
            + log_q_y.mean() - (log_mix_y.mean() - np.log(2))) / 2


def gmm_kl(gmm_p, gmm_q, sample_count=500):
    x = gmm_p.sample(sample_count)[0]
    log_p_x = gmm_p.score_samples(x)
    log_q_x = gmm_q.score_samples(x)
    return log_p_x.mean() - log_q_x.mean()


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


class MusicSimModel:
    """
    Similarity model that is used for storage and prediction.
    """

    def __init__(self, gmm_parameters, classes, class_names, svc):
        self.gmm_parameters = gmm_parameters
        self.classes = classes
        self.class_names = class_names
        self.svc = svc

        # initialize gmms in the event that the model is
        # going to be used for prediction.
        self.gmms = None
        self.kernel_func = None

    @staticmethod
    def create_kernel(gamma: float = 0.1):
        def kernel(js: float):
            return np.exp(-gamma * js)

        return np.vectorize(kernel)

    def load(self):
        """
        Loads gaussian mixture models from `gmm_parameters` into memory.
        """
        self.gmms = list(map(init_gmm, self.gmm_parameters))
        self.kernel = self.create_kernel()

    def predict_file(self, x, n_comp=20):
        """
        Predicts the class of the given list of samples.

        :param x: list of strings, where each string is the path to a file.
        :param n_comp: number of mfcc components to use, should be equal to
            what has been used for training the dataset.
        :return: list of tuples, where each tuple has the probability that that
            sample is of that class and the predicted class.
        """
        if self.gmms is None:
            self.load()

        samples = np.empty((len(x), len(self.gmm_parameters)))
        for i, sample in enumerate(x):
            mfcc = convert_to_mfcc(sample)
            mfcc = np.nan_to_num(mfcc.real).T

            gmm = mixture.GaussianMixture(
                n_components=n_comp, covariance_type='full')
            gmm.fit(mfcc)

            sims = list(map(lambda x: gmm_js(x, gmm), self.gmms))
            samples[i, :] = self.kernel(sims).reshape(1, -1)

        predict = self.svc.predict_proba(samples)

        results = []
        for sample in predict:
            max_prob_index = np.argmax(sample)
            results.append((sample[max_prob_index], max_prob_index))

        return results
