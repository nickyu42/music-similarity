### Music Similarity

The goal of this project is to research how similar Shōnen/ Shōjo Anime openings are.  
The system uses song signal level features to check the similarity of songs. In short, a song is modelled as a [Gaussian Mixture Model (GMM)](https://en.wikipedia.org/wiki/Mixture_model) where the "similarity" of two songs is measured as the "distance" between the GMMs. Multiple types of distance measures can be used, but most examples here use the [Jenson-Shannon Divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence).

A full example is given in this [notebook](notebooks/train.ipynb).  
It includes an example on training and using the music similarity model.

Note that this is a toy project and is more on the experimental side.

References  
- Mandel, M. I., & Ellis, D. P. (2005). Song-level features and support vector machines for music classification.  
- Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2011). Using Mutual Proximity to Improve Content-Based Audio Similarity. In ISMIR (Vol. 11, pp. 79-84).  
- Jensen, J. H., Ellis, D. P., Christensen, M. G., & Jensen, S. H. (2007). Evaluation distance measures between gaussian mixture models of mfccs.  
- Tzanetakis, G. (n.d.). Icmr2014_tutorial_2_per_page.pdf. http://www.marsyas.info/icmr2014_tutorial_2_per_page.pdf
