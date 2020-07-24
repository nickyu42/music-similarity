"""
Module for training and storing a model.

The model consists of the SVM parameters, but also the gmm parameters 
of each song in the initial dataset.
"""
import pickle
import pathlib
from typing import List, Tuple
from collections import namedtuple

import numpy as np
from sklearn import svm

from processing import convert_to_mfcc
from classification import gmm_js, fit_gmm, init_gmm


def compute_gmm_parameters(songs: List[str]) -> List[np.ndarray]:
    """
    Computes gaussian mixture parameters for each given song.

    :param songs: list of paths to each song.
    :return: list of gmm parameters for each song.
    """
    samples = []

    for song_path in songs:
        mfccs = convert_to_mfcc(song_path, frames=3000)
        samples.append(fit_gmm(mfccs))

    return samples


def compute_gram_matrix(samples_gmm: List[np.ndarray]) -> np.ndarray:
    """
    Precomputes a gramm matrix from samples in the form of gmm parameters.

    :param samples_gmm: list of songs represented as gmm parameters.
    :return: np.ndarray of js distance between each song.
    """
    # create gram matrix of #songs . #songs
    song_count = len(samples_gmm)
    gram_matrix = np.zeros((song_count, song_count))

    # precompute d_js
    gmms = map(init_gmm, samples_gmm)
    for i, x in enumerate(gmms):
        for j, y in enumerate(gmms):

            # only compute half of the matrix, because js is symmetric
            # the other half will be copied
            if j <= i:
                gram_matrix[i, j] = gmm_js(x, y)
            else:
                gram_matrix[i, j] = gram_matrix[j, i]

    return gram_matrix


def train(songs: List[str], classes: List[int]) -> Tuple[svm.SVC, List[np.ndarray]]:
    """
    Fits an SCV on the given songs.

    :param songs: list of paths to each song.
    :param classes: the corresponding class for each song, must be in 
        [1, unique_classes].
    :return: SVC and gmm_parameters for storing.
    """
    print('Computing Gaussian Mixture Models and svc gramm matrix')
    gmm_parameters = compute_gmm_parameters(songs)
    gram_matrix = compute_gram_matrix(gmm_parameters)

    # apply custom rbf kernel
    def kernel(js: float, gamma: float = 0.1):
        return np.exp(-gamma * js)

    # add possible speedup?
    vectorized_kernel = np.vectorize(kernel)

    print('Applying rbf')
    gram_matrix = vectorized_kernel(gram_matrix)

    svc = svm.SVC(kernel='precomputed')

    print('Training SVC')
    svc.fit(gram_matrix, classes)

    return svc, gmm_parameters


MusicSimModel = namedtuple(
    'MusicSimModel', ('gmm_parameters', 'classes', 'class_names', 'svc')
)


def train_and_store(songs: List[str], classes: List[int], class_names: List[str], path: str, filename: str):
    """
    Creates an instance of `MusicSimModel` from the given dataset
    and stores it as a pickled python object.

    :param songs: list of paths to each song.
    :param classes: the corresponding class for each song, must be in 
        [1, unique_classes].
    :param class_names: list of class identifiers.
    :param path: the path to store the model.
    :param filename: the name to give the new file.
    """
    path = pathlib.Path(path)

    if not path.is_dir:
        raise ValueError(f'Given path {path} is not a directory.')

    if len(songs) != len(classes):
        raise ValueError(
            'number of samples is not equal to list of corresponding classes.')

    for song_path in songs:
        if not pathlib.Path(song_path).is_file:
            raise ValueError(
                f'Song {song_path} does not exist or is not a file.')

    print('Starting training of model')
    print('--------------------------')
    svc, gmm_parameters = train(songs, classes)

    model = MusicSimModel(gmm_parameters, classes, class_names, svc)

    try:
        with (path / filename).open('wb') as f:
            pickle.dump(model, f)

        print('-------------------------')
        print('Succesfully trained model')
        print(f'Model was saved to {path.resolve()}')
    except Exception as e:
        print('Exception occured during training:')
        print(e)
