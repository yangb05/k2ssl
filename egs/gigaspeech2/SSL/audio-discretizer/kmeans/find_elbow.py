from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt


def load_embed(embed_files):
    cut_embeds = {}
    for file in embed_files.split(','):
        cut_embeds.update(np.load(file, allow_pickle=True).item())
    embeds = np.concatenate(list(cut_embeds.values()), axis=0)
    return embeds


def compute(embed_files):
    distortions = []
    inertias = []
    K = range(1000, 10000, 1000)
    for k in K:
        # Building and fitting the model
        X = load_embed(embed_files)  
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(X)
        distortions.append((k, sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]))
        inertias.append((k, kmeanModel.inertia_))
    distortions = zip(*distortions)
    inertias = zip(*inertias)
    return distortions, inertias


def plot(distortions, inertias):
    plt.plot(distortions[0], distortions[1], 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.savefig('/mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/audio-discretizer/kmeans/distortions.png')
    
    plt.plot(inertias[0], inertias[1], 'rx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.savefig('/mgData2/yangb/icefall-ssl/egs/gigaspeech2/SSL/audio-discretizer/kmeans/inertias.png')


def run(embed_files):
    distortions, inertias = compute(embed_files)
    plot(distortions, inertias)


if __name__ == '__main__':
    embed_files = "/data_a100/userhome/yangb/data/fbank/librispeech_embed_train-clean-100.npy"
    # embed_files = "/data_a100/userhome/yangb/data/fbank/librispeech_embed_train-clean-100.npy,/data_a100/userhome/yangb/data/fbank/librispeech_embed_train-clean-360.npy,/data_a100/userhome/yangb/data/fbank/librispeech_embed_train-other-500.npy"
    run(embed_files)

    
