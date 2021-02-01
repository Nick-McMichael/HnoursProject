import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import train_test_split

x, y = pd.read_csv(r'C:\Users\Nick\Desktop\Project Data\XY.csv', delimiter=";")
data_csv = pd.read_csv(r'C:\Users\Nick\Desktop\Project Data\Data_Test.csv', delimiter=";")
print(data_csv)

df1 = data_csv.replace(np.nan, 0, regex=True)
X_std = MinMaxScaler().fit_transform(df1)
dataframe = pd.DataFrame(X_std, columns = df1.columns)

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


train, test = train_test_split(dataframe, test_size=0.2, random_state=42, shuffle=True)

# plt.scatter(train, alpha=0.8)
# plt.title('Scatter plot')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()