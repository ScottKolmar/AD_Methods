import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from sklearn.metrics import pairwise, mean_squared_error
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
csv_file = r'C:\Users\skolmar\PycharmProjects\Modeling\AD_Methods\Datasets\g298atom_desc.csv'
df = pd.read_csv(csv_file, header=0, index_col=0)
df = df.sample(1000, random_state=42)

# Split into train and test
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=True, random_state=11)

# convert dataframes to numpy arrays
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

# Calculate nearest neighbors for each compound in test set
neighbors = NearestNeighbors(n_neighbors=5, metric='euclidean')
neighbors.fit(X=X_train)
dists, indices = neighbors.kneighbors(X=X_test_np, n_neighbors=5, return_distance=True)

# Define function
def calc_ave_cos_sim(x):
    i = np.where(X_test_np == x)[0][0]
    nearest_vecs = [X_train_np[ind] for ind in indices[i]]
    ave_cos_sim = np.mean([pairwise.cosine_similarity(x.reshape(1,-1), y.reshape(1,-1)) for y in nearest_vecs])
    return ave_cos_sim

# Get values by applying function to individual X_test elements
first_list = []
for i,elem in enumerate(X_test_np):
    compound = np.array(X_test.iloc[i])
    first_list.append(calc_ave_cos_sim(compound))

# Map method
ave_cos_sims = np.array(list(map(calc_ave_cos_sim, X_test_np)))
second_list = []
for i,elem in enumerate(ave_cos_sims):
    second_list.append(elem)

print(first_list == second_list)

knn = KNeighborsRegressor(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
mse = (y_test - y_pred)**2
knn_corr, knn_p = pearsonr(ave_cos_sims, mse)
print(f'kNN Corr: {knn_corr}')
print(f'kNN p-val: {knn_p}')

# plt.plot(ave_cos_sims, mse, 'ro')
# plt.xlabel('Cosine Similarity')
# plt.ylabel('Squared error')
# plt.title('Knn')
# plt.show()

# rf = RandomForestRegressor()
# rf.fit(X_train, y_train)
# rf_pred = rf.predict(X_test)
# mse_rf = (y_test - rf_pred)**2

# plt.plot(ave_cos_sims, mse_rf, 'bo')
# plt.xlabel('Cosine Similarity')
# plt.ylabel('Squared error')
# plt.title('RF')
# plt.show()