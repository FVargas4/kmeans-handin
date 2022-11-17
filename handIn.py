import matplotlib.pyplot as plt
import numpy as num
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# create dataset  (y (labels) are not used)
# X, y = make_blobs(
#    n_samples=150, n_features=2,
#    centers=3, cluster_std=0.5,
#    shuffle=True, random_state=0
# )

aux = pd.read_csv('CC_GENERAL.csv')

X = num.array(aux)


print("Data X")   
print(X)

# plot
plt.scatter(
   X[:, 0], X[:, 1],
   c='black', marker='o',
   edgecolor='white', s=50
)
plt.show()



km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X)
print(y_km)



# plot the 3 clusters
plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='red',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='green',
    marker='o', edgecolor='black',
    label='cluster 2'
)

plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='blue',
    marker='v', edgecolor='black',
    label='cluster 3'
  )

# plt.scatter(
#     X[y_km == 3, 0], X[y_km == 3, 1],
#     s=50, c='yellow',
#     marker='h', edgecolor='black',
#     label='cluster 4'
#   )

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='X',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()