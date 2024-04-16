import os
import sys

import joblib
import numpy as np

kmeans_model_path = sys.argv[1]
kmeans_model_dir = os.path.dirname(kmeans_model_path)

model = joblib.load(kmeans_model_path)

codebook = model.cluster_centers_

print(codebook.shape)

codebook_path = os.path.join(kmeans_model_dir, "codebook.npy")
np.save(codebook_path, codebook)
