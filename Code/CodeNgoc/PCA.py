import numpy as np
import pandas as pd
import time
# For plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline
# 'exec(%matplotlib inline)'
from IPython import get_ipython

ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
# PCA
from sklearn.decomposition import PCA

# TSNE
# from sklearn.manifold import TSNE
# UMAP
# import umap

train = pd.read_csv('sign_mnist_train.csv')
train.head()

# Setting the label and the feature columns
y = train.loc[:, 'label'].values
x = train.loc[:, 'pixel1':].values
print(np.unique(y))

# Appling PCA
start = time.time()
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
print('Duration: {} seconds'.format(time.time() - start))
principal = pd.DataFrame(data=principalComponents,
                         columns=['principal component 1', 'principal component 2', 'principal component 3'])
principal.shape

# Plotting PCA 2D
plt.style.use('dark_background')
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y, cmap='gist_rainbow')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(20)).set_ticks(np.arange(20))
plt.title('Visualizing sign-language-mnist through PCA', fontsize=20);
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Plotting PCA 2D
plt.style.use('dark_background')
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y, cmap='gist_rainbow')
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(24)).set_ticks(np.arange(24))
plt.title('Visualizing sign-language-mnist through PCA', fontsize=24);
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
