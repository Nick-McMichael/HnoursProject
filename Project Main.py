import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from matplotlib import *
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
from scipy import stats
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.manifold import TSNE

x, y = pd.read_csv(r'C:\Users\Nick\Desktop\Project Data\XY.csv', delimiter=";")
data_csv = pd.read_csv(r'C:\Users\Nick\Desktop\Project Data\Data_Test.csv', delimiter=";")

df1 = data_csv.replace(np.nan, 0, regex=True)
X_std = MinMaxScaler().fit_transform(df1)
dataframe = pd.DataFrame(X_std, columns = df1.columns)\

cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

pca = sklearnPCA(n_components=4)
pca.fit_transform(dataframe)
print('Explained varaiance ratio: ', pca.explained_variance_ratio_)

#Explained variance
pca = sklearnPCA().fit(dataframe)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


db = DBSCAN(
 eps = .2,
 metric="euclidean",
 min_samples = 5,
 n_jobs = -1)
clusters = db.fit_predict(dataframe)

print(db.labels_)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

X_2D = TSNE(n_components=2, perplexity=30, n_iter=400).fit_transform(df1) # collapse in 2-D space for plotting
for i in set(db.labels_):
    print('class {}: number of points {:d}'.format(i, np.sum(db.labels_==i)))

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# ax.scatter(X_2D[pos, 0], X_2D[pos, 1], c=[[0.8, 0.4, 0.4],], marker='x', s=80, label='Positive')
for i in set(db.labels_):
    if i == -1: 
        #outlier according to dbscan
        ax.scatter(X_2D[db.labels_==i, 0], X_2D[db.labels_==i, 1], c='r', s=8, label='DBSCAN Outlier')
    else:
        ax.scatter(X_2D[db.labels_==i, 0], X_2D[db.labels_==i, 1], s=8, c=[[0.2, 0.3, max(i*0.2 + 0.4, 1)],],
                                                                           label='DBSCAN class {}'.format(i))
plt.axis('off')
plt.legend()
plt.show()     

# Standard library imports
from collections import Counter, defaultdict
import time
import os

# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (auc, average_precision_score, 
                              roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.base import BaseEstimator

# plt.rcParams.update({'font.size': 12})


def labels_from_DBclusters(db):
    """
    Returns labels for each point for "outlierness", based on DBSCAN results.
    The higher the score, the more likely the point is an outlier, based on its cluster membership
    
    - dbscan label -1 (outliers): highest score of 1
    - largest cluster gets score 0  
    - points belonging to clusters get a score that is higher when the cluster size is smaller
    
    db: a fitted DBscan instance
    Returns: labels (similar to "y_predicted", but the values merely reflect a ranking)
    """
    labels = np.zeros(len(db.labels_))
    
    # make a list of tuples: (i, num points in i) for i in db.labels_
    label_counts = [(i, np.sum(db.labels_==i)) for i in set(db.labels_) - set([-1])]
    label_counts.sort(key=lambda x : -x[1]) # sort by counts per class, descending
    
    # assign the labels. Those points with label =-1 get highest label (equal to number of classes -1) 
    labels[db.labels_== -1] = len(set(db.labels_)) - 1
    for i, (label, label_count) in enumerate(label_counts):
        labels[db.labels_==label] = i
        
    # Scale the values between 0 and 1
    labels = (labels - min(labels)) / (max(labels) - min(labels))
    return(labels) 

   
def dbscan_outlier_pred(X, epsilon=0.3, min_samples=10, **kwargs):
    db = DBSCAN(eps=epsilon, min_samples=min_samples, **kwargs)
    db.fit(X)
    return labels_from_DBclusters(db)

# Run this for various min_samples and epsilon
def dbscan_scan(X, min_samples_list=(5, 10), epsilon_list=(1, 2, 5, 10), random_state=None):
    """ A scan function for DBSCAN. Iterates over min_samples_list and epsilon_list
    random_state (None or int) : if int, this will be set as a random seed and data will be shuffled
    """
    dbscan_dict = defaultdict(list)
    if random_state:
        np.random.seed(random_state)
        idx = np.random.choice(len(y), len(y), replace=False)
        X = X[idx, :]
        y = y[idx]

    for min_samples in min_samples_list:
        for epsilon in epsilon_list:            
            print('calculating result for epsilon {}, min_samples {}...'.format(
            epsilon, min_samples))
            
            # dbscan_outlier_pred returns 1 for the outlier class, and 0 for the main "inlier" class
            y_predicted = dbscan_outlier_pred(X, epsilon=epsilon, min_samples=min_samples) 
            auc_db = roc_auc_score(y_true=y, y_score=y_predicted)
            pr_db = average_precision_score(y_true=y, y_score=y_predicted)
            num_clusters = len(set(y_predicted))
            # precision_outlier_class = y[y_predicted == 1].mean()
            size_outlier_cluster = np.sum(y_predicted == 1)

            # store results in a DataFrame
            
            for k, v in (('epsilon', epsilon), ('min_samples', min_samples), #('AUC', auc_db), 
                         ('num_clusters', num_clusters), ('size_outlier_cluster', size_outlier_cluster), 
                         #('AP', pr_db), 
                         #('precision_outlier_class', precision_outlier_class),
                         ('random_state', random_state)):
                dbscan_dict[k].append(v)
        dbscan_results_df = pd.DataFrame.from_dict(dbscan_dict)

    return dbscan_results_df

dbscan_results_df = dbscan_scan(df1)