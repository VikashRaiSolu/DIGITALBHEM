from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Column names for the dataset
names = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width', 'Class']

# Load the dataset
dataset = pd.read_csv("8.csv", names=names)

# Separate features and target labels
X = dataset.iloc[:, :-1].values
label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = np.array([label[c] for c in dataset.iloc[:, -1]])

# Plot settings
plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'lime', 'black'])

# Real plot
plt.subplot(1, 3, 1)
plt.title('Real')
plt.scatter(X[:, 2], X[:, 3], c=colormap[y])

# K-Means plot
model = KMeans(n_clusters=3, random_state=0).fit(X)
plt.subplot(1, 3, 2)
plt.title('KMeans')
plt.scatter(X[:, 2], X[:, 3], c=colormap[model.labels_])

print('The accuracy score of K-Mean: ', metrics.accuracy_score(y, model.labels_))
print('The Confusion matrix of K-Mean:\n', metrics.confusion_matrix(y, model.labels_))

# GMM plot
gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
y_cluster_gmm = gmm.predict(X)
plt.subplot(1, 3, 3)
plt.title('GMM Classification')
plt.scatter(X[:, 2], X[:, 3], c=colormap[y_cluster_gmm])

print('The accuracy score of EM: ', metrics.accuracy_score(y, y_cluster_gmm))
print('The Confusion matrix of EM:\n', metrics.confusion_matrix(y, y_cluster_gmm))

plt.show()
