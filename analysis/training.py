"""
Module for training and storing a model.

The model consists of the SVM parameters, but also the gmm parameters 
of each song in the initial dataset.
"""
import pickle
import pathlib
import threading
from multiprocessing import Pool, freeze_support
from typing import List, Tuple
from collections import namedtuple

import numpy as np
from sklearn import svm
from tqdm.notebook import tqdm

from processing import convert_to_mfcc
from classification import gmm_js, fit_gmm, init_gmm, MusicSimModel


def _gmm_worker(songs, worker_id=0):
    """
    Worker for computing gmms.

    :param songs: list of paths to each song.
    :param worker_id: worker to display in progress bar.
    """
    gmms = []

    # HACK: workaround for multiprocessing/tqdm in jupyter
    print(' ', end='', flush=True)

    for song in tqdm(songs, position=worker_id, desc=f'worker {worker_id}'):
        gmms.append(fit_gmm(convert_to_mfcc(song, frames=30000)))

    return gmms


def compute_gmm_parameters(songs: List[str], processes: int = 1) -> List[np.ndarray]:
    """
    Computes gaussian mixture parameters for each given song.

    :param songs: list of paths to each song.
    :param processes: number of parallel processes to run.
    :return: list of gmm parameters for each song.
    """
    if processes < 0 or type(processes) is not int:
        raise ValueError('Number of processes must be a postive integer.')

    print(f'Computing Gaussian Mixture Models with procceses={processes}')

    if processes == 1:
        return _gmm_worker(songs)

    freeze_support()
    
    pool = Pool(processes=processes, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))

    # floor the partition size
    partition_size = int(len(songs) / processes)
    results = []

    for i in range(processes - 1):
        partition = songs[i*partition_size:i*partition_size + partition_size]
        results.append(pool.apply_async(_gmm_worker, (partition, i)))

    # add final partition
    # add remaining songs in the event that len(songs) / processes is a fraction
    final_partition = (processes - 1)*partition_size
    results.append(
        pool.apply_async(_gmm_worker, (songs[final_partition:], processes - 1))
    )

    all_gmms = []

    # join all futures in order
    for result in results:
        result.wait()
        all_gmms.extend(result.get())

    return all_gmms


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
    gmms = list(map(init_gmm, samples_gmm))

    print('Computing svc gramm matrix')
    for i, x in tqdm(enumerate(gmms)):
        for j, y in enumerate(gmms):

            # only compute half of the matrix, because js is symmetric
            # the other half will be copied
            if j >= i:
                gram_matrix[i, j] = gmm_js(x, y)
            else:
                gram_matrix[i, j] = gram_matrix[j, i]

    return gram_matrix


def train(songs: List[str], classes: List[int], processes: int = 1) -> Tuple[svm.SVC, List[np.ndarray]]:
    """
    Fits an SCV on the given songs.

    :param songs: list of paths to each song.
    :param classes: the corresponding class for each song, must be in 
        [1, unique_classes].
    :return: SVC and gmm_parameters for storing.
    """
    gmm_parameters = compute_gmm_parameters(songs, processes)
    gram_matrix = compute_gram_matrix(gmm_parameters)

    # apply custom rbf kernel
    def kernel(js: float, gamma: float = 0.1):
        return np.exp(-gamma * js)

    # add possible speedup?
    vectorized_kernel = np.vectorize(kernel)

    print('Applying rbf')
    gram_matrix = vectorized_kernel(gram_matrix)

    svc = svm.SVC(kernel='precomputed', probability=True)

    print('Training SVC')
    svc.fit(gram_matrix, classes)

    return svc, gmm_parameters


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


def main():
    s = ['data/songs/Dragon Ball.wav', 'data/songs/Akagami no Shirayuki-hime.wav']
    c = [1, 2]

    svc, gmm_parameters = train(s, c)

    # TODO: check why classes are flipped while similarities are correct

    model = MusicSimModel(gmm_parameters, c, ['shounen', 'shoujo'], svc)

    model.load()
    print(model.predict_file(
        ['data/songs/Dragon Ball.wav', 'data/songs/Akagami no Shirayuki-hime.wav']))


if __name__ == "__main__":
    main()
