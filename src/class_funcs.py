### IMPORTS ###
from math import dist
from re import M
import pandas as pd
import numpy as np
import os
import pprint as pp
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Math imports
from scipy.stats import pearsonr, linregress

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise, mean_squared_error, roc_auc_score
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

### FUNCTIONS ###

####################
# DISTANCE FUNCTIONS
####################

# class Distance():
#     def __init__(self):
#         self.name = None
#         self.distance = None


def euclidean(v1, v2):
    name = 'euclidean'
    distance = np.sqrt(np.sum((v1-v2)**2))
    return distance, name

def cityblock(v1, v2):
    name = 'cityblock'
    distance = np.sum(np.abs(v1-v2))
    return distance, name

def minkowski(v1, v2, n=2):
    name = 'minkowski'
    distance = np.sum(np.abs(v1-v2)**n)**(1/n)
    return distance, name

def chebyshev(v1, v2):
    name = 'chebyshev'
    distance = np.max(np.abs(v1-v2))
    return distance, name

def sorensen(v1, v2):
    name = 'sorensen'
    A = np.sum(np.abs(v1-v2))
    d = len(v1)
    distance = A/np.sum(v1+v2)
    return distance, name

def gower(v1, v2):
    name = 'gower'
    A = np.sum(np.abs(v1-v2))
    d = len(v1)
    distance = A/d
    return distance, name

def sorgel(v1, v2):
    name = 'sorgel'
    A = np.sum(np.abs(v1-v2))
    d = len(v1)
    distance = A/np.sum(np.maximum(v1,v2))
    return distance, name

def kulczynski(v1, v2):
    name = 'kulczynski'
    A = np.sum(np.abs(v1-v2))
    d = len(v1)
    distance = A/np.sum(np.minimum(v1,v2))
    return distance, name

def canberra(v1,v2):
    name = 'canberra'
    A = np.sum(np.abs(v1-v2))
    d = len(v1)
    distance = np.sum(np.abs(v1,v2)/(v1+v2))
    return distance, name

def lorentzian(v1,v2):
    name = 'lorentzian'
    A = np.sum(np.abs(v1-v2))
    d = len(v1)
    distance = np.sum(np.log(1+np.abs(v1-v2)))
    return distance, name

def czekanowski(v1,v2):
    name = 'czekanowski'
    A = np.sum(np.abs(v1-v2))
    distance = A/np.sum(v1+v2)
    return distance, name

def ruzicka(v1,v2):
    name = 'ruzicka'
    distance = 1-np.sum(np.minimum(v1,v2))/np.sum(np.maximum(v1,v2))
    return distance, name

def tanimoto(v1,v2):
    name = 'tanimoto'
    distance = np.sum(np.maximum(v1,v2)-np.minimum(v1,v2))/np.sum(np.maximum(v1,v2))
    return distance, name

def cosine_sim(v1,v2):
    name = 'cosine_similarity'
    distance = pairwise.cosine_similarity(v1,v2)
    return distance, name

######################
# DATASET FUNCTIONS
######################

class DataSet():

    def __init__(self, csv, size, **kwargs):
        self.csv = csv
        self.dataset_name = csv.split('\\')[-1].split('_')[0]
        self.df = pd.read_csv(csv, header=0, index_col=0).sample(size, **kwargs)
        self.X = self.df.iloc[:,:-1]
        self.y = self.df.iloc[:,-1]

        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

        self.X_train_np = []
        self.X_test_np = []
        self.y_train_np = []
        self.y_test_np = []
        
        self.indices = []
        self.distances = []

        self.score_dicts = {}

        self.ad_measures = {
            'average_cosine_similarity': [],
            'average_euclidean': [],
            'average_cityblock': [],
            'average_minkowski': [],
            'average_chebyshev': [],
            'average_sorensen': [],
            'average_gower': [],
            'average_sorgel': [],
            'average_kulczynski': [],
            'average_canberra': [],
            'average_lorentzian': [],
            'average_czekanowski': [],
            'average_ruzicka': [],
            'average_tanimoto': [],
            'length_of_means': [],
            'average_mahalanobis': []
            }

    def split(self, **kwargs):
        """ Splits into train and test set."""

        X_train, X_test, y_train, y_test= train_test_split(self.X, self.y, **kwargs)

        self.X_train.append(X_train)
        self.X_test.append(X_test)
        self.y_train.append(y_train)
        self.y_test.append(y_test)

        self.X_train_np.append(X_train.to_numpy())
        self.X_test_np.append(X_test.to_numpy())
        self.y_train_np.append(y_train.to_numpy())
        self.y_test_np.append(y_test.to_numpy())

        return None

    def get_neighbors(self, n_neighbors, **kwargs):
        """ Calculates the n nearest neighbors for every entry in the test set."""

        for i in range(len(self.X_train)):
            neighbors = NearestNeighbors(n_neighbors=n_neighbors, **kwargs)
            neighbors.fit(X=self.X_train_np[i])
            dists, indices = neighbors.kneighbors(X=self.X_test_np[i], n_neighbors=5, return_distance=True)
            self.indices.append(indices)
            self.distances.append({neighbors.effective_metric_:dists})

        return None

##############################
# AD MEASURES
##############################

    def calculate_average_cosine_similarity(self):
        """ Calculates the average cosine similarity of n nearest neighbors for every entry in the test set."""

        for i_1 in range(len(self.X_train)):
            ave_cos_list = []
            for i_2 in range(len(self.X_test_np[i_1])):
                compound = np.array(self.X_test[i_1].iloc[i_2])
                compound_index = np.where(self.X_test_np[i_1] == compound)[0][0]
                nearest_vectors = [self.X_train_np[i_1][neighbor_index] for neighbor_index in self.indices[i_1][i_2]]
                ave_cos_sim = np.mean([pairwise.cosine_similarity(compound.reshape(1,-1), neighbor_vec.reshape(1,-1)) for neighbor_vec in nearest_vectors])
                ave_cos_list.append(ave_cos_sim)
            
            self.ad_measures['average_cosine_similarities'].append(ave_cos_list)
        
        return None
    
    def calculate_average_distance(self):
        """ Calculates the mean distance of each test set entry to its n nearest neighbors."""

        # Loop through each test set
        for i in range(len(self.X_test)):

            # Get corresponding array of neighbor indices
            neighbor_indices_array = self.indices[i]

            # Empty list of distances for a test set
            mean_euc_distances = []

            # Loop through each compound in np array of test set
            for compound_index in range(len(self.X_test_np[i])):

                # Get compound vector
                compound = self.X_test_np[i][compound_index]

                # Get neighbor vectors
                neighbor_indices = neighbor_indices_array[compound_index]
                neighbor_vecs = [self.X_train_np[i][neighbor_idx] for neighbor_idx in neighbor_indices]

                # Calculate distances and add to test set list of distances
                mean_distance = np.mean([pairwise.euclidean_distances(compound.reshape(1,-1), neighbor_vec.reshape(1,-1)) for neighbor_vec in neighbor_vecs])

                mean_euc_distances.append(mean_distance)
            
            # Append test set distances to dataset list of distances
            self.ad_measures['average_euclidean_distances'].append(mean_euc_distances)

        return None
    
    def calculate_all_distances(self):
        """ Calculates all mean distance measures for each test set entry to its n nearest neighbors."""

        # Loop through each distance measure
        for function in [euclidean, cityblock, minkowski, chebyshev, sorensen, gower, sorgel, kulczynski, canberra, lorentzian, czekanowski, ruzicka, tanimoto, cosine_sim]:
            
            # Loop through each test set
            for i in range(len(self.X_test)):
        
                # Get corresponding array of neighbor indices
                neighbor_indices_array = self.indices[i]

                # Empty list of distances for a test set
                mean_distances = []

                # Loop through each compound in np array of test set
                for compound_index in range(len(self.X_test_np[i])):
                    
                    # Get compound vector
                    compound = self.X_test_np[i][compound_index]

                    # Get neighbor vectors
                    neighbor_indices = neighbor_indices_array[compound_index]
                    neighbor_vecs = [self.X_train_np[i][neighbor_idx] for neighbor_idx in neighbor_indices]

                    # Calculate distances and add to test set list of distances
                    dist_tuples = [function(v1 = compound.reshape(1,-1), v2 = neighbor_vec.reshape(1,-1)) for neighbor_vec in neighbor_vecs]
                    mean_distance = np.mean([x[0] for x in dist_tuples])
                    name = dist_tuples[0][1]
                    
                    mean_distances.append(mean_distance)
                
                # Append test set distances to dataset list of distances
                self.ad_measures[f'average_{name}'].append(mean_distances)

        return None

    def calculate_length_mean_vector(self):
        """ """
        # Loop through each test set
        for i in range(len(self.X_test)):

            # Get corresponding array of neighbor indices
            neighbor_indices_array = self.indices[i]

            # Empty list of distances for a test set
            length_of_mean_vecs = []

            # Loop through each compound in np array of test set
            for compound_index in range(len(self.X_test_np[i])):

                # Get compound vector
                compound = self.X_test_np[i][compound_index]

                # Get neighbor vectors
                neighbor_indices = neighbor_indices_array[compound_index]
                neighbor_vecs = [self.X_train_np[i][neighbor_idx] for neighbor_idx in neighbor_indices]

                # Calculate lengths of each mean vector and add to test set list
                distances = [pairwise.euclidean_distances(compound.reshape(1,-1), neighbor_vec.reshape(1,-1)) for neighbor_vec in neighbor_vecs]
                mean_vector = np.mean(distances, axis=0)
                length = np.linalg.norm(mean_vector)
                length_of_mean_vecs.append(length)

            # Append test set list of lengths to dataset list
            self.ad_measures['length_of_means'].append(length_of_mean_vecs)

            return None

#########################
# Algorithm predictions
#########################

    def pred_kNN(self, n_neighbors, **kwargs):
        """ Generates predictions using kNN, calculates the prediction error, and calculates
        the Pearson Correlation Coefficient between the prediction error and the AD measures."""

        self.score_dicts['kNN'] = {
            'Algorithm': {
                'Estimators': [],
                'Predictions': [],
                'Squared_error': []
                },
            'Results':{
                'average_cosine_similarities': [],
                'average_euclidean_distances': [],
                'length_of_means': [],
                'mahalanobis_distances': []
                }
            }

        # Loop through each training set
        for i in range(len(self.X_train)):
            knn = KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs)
            knn.fit(self.X_train[i], self.y_train[i])
            self.score_dicts['kNN']['Algorithm']['Estimators'].append(knn.get_params())

            y_pred = knn.predict(self.X_test[i])
            self.score_dicts['kNN']['Algorithm']['Predictions'].append(y_pred)

            squared_error = (self.y_test[i] - y_pred)**2
            self.score_dicts['kNN']['Algorithm']['Squared_error'].append(squared_error)
        
        return None
    
    def pred_random_forest(self, **kwargs):
        """ Generates predictions using Random Forest, calculates the prediction error, and calculates
        the Pearson Correlation Coefficient between the prediction error and the AD measures."""

        self.score_dicts['RF'] = {
            'Algorithm': {
                'Estimators': [],
                'Predictions': [],
                'Squared_error': [],
                'Tree_predictions': []
            },
            'Results': {
                'average_cosine_similarities': [],
                'average_euclidean_distances': [],
                'length_of_means': [],
                'mahalanobis_distances': [],
                'pred_std': []
            }
        }

        # Loop through each training set
        for i in range(len(self.X_train)):

            # Fit random forest and create tree dictionary
            rf = RandomForestRegressor(**kwargs)
            rf.fit(self.X_train[i], self.y_train[i])
            self.score_dicts['RF']['Algorithm']['Estimators'].append(rf.get_params())
            self.score_dicts['RF']['Algorithm']['Tree_predictions'].append({})

            # Predit with each estimator (tree)
            for each_tree in range(rf.n_estimators):
                preds = rf.estimators_[each_tree].predict(self.X_test[i])
                self.score_dicts['RF']['Algorithm']['Tree_predictions'][i][f'tree_{each_tree}'] = preds

            # Predict with ensemble
            y_pred = rf.predict(self.X_test[i])
            self.score_dicts['RF']['Algorithm']['Predictions'].append(y_pred)

            # Calculate squared error for each prediction
            squared_error = (self.y_test[i] - y_pred)**2
            self.score_dicts['RF']['Algorithm']['Squared_error'].append(squared_error)

        return None
    
#####################
# GET RESULTS
#####################
    
    def calculate_tree_std(self):
        """ Calculates the standard deviation for a single test set entry from the ensemble of single
        tree predictions in a random forest estimator."""

        # Loop through each set of results
        for i in range(len(self.X_train)):
            
            # Calculate standard error for each test set entry mean prediction using tree prediction arrays
            tree_df = pd.DataFrame.from_dict(data=self.score_dicts['RF']['Tree_predictions'][i], orient='columns')
            pred_std = tree_df.std(axis=1)
            self.score_dicts['RF']['Results']['pred_std'].append(pred_std)

        return None
    
    def calculate_stats(self, algorithm, ad_measure, print):
        """ Calculates pearson correlation coefficient, slope, and R2 for the
        provided applicability domain measure, for the provided algorithm."""

        # Set algorithm dictionary
        alg_dict = self.score_dicts[algorithm]

        # Loop through each set of results
        for i in range(len(self.X_train)):

            # Set dependent variable
            pred_error = alg_dict['Algorithm']['Squared_error'][i]

            # Set ad_measure (indendependent variable)
            if ad_measure not in self.ad_measures.keys():
                return f'AD measure must be one of {self.ad_measures.keys()}'
            else:
                ad_measure_array = self.ad_measures[ad_measure][i]

            # Calculate pearson r
            corr, corr_p = pearsonr(ad_measure_array, pred_error)

            # Calculate linear fit
            lin_result = linregress(ad_measure_array, pred_error)

            # Store results
            result_dict = {
                'Pearson_correlation': corr,
                'Pearson_p_val': corr_p,
                'Slope': lin_result.slope,
                'Stderr': lin_result.stderr,
                'p_val': lin_result.pvalue,
                'R2': lin_result.rvalue**2
            }
            alg_dict['Results'][ad_measure].append(result_dict)

            # Print results
            if print:
                printer = pp.PrettyPrinter(indent=4)
                printer.pprint(alg_dict['Results'][ad_measure])

            return None

    def plot_correlations(self, algorithm, ad_measure, out_path):
        """ Plots correlations between prediction errors and AD measures."""

        # Loop through all results
        for i in range(len(self.ad_measures[ad_measure])):

            # Get ad measure array and prediction error array
            ad_measure_array = self.ad_measures[ad_measure][i]
            pred_error = self.score_dicts[algorithm]['Algorithm']['Squared_error'][i]

            # Get result stats
            result_dict = self.score_dicts[algorithm]['Results'][ad_measure][i]
            pearson_corr = result_dict['Pearson_correlation']
            pearson_p = result_dict['Pearson_p_val']
            slope = result_dict['Slope']
            stderr = result_dict['Stderr']
            p_val = result_dict['p_val']
            r2 = result_dict['R2']

            label = f'Pearson R: {pearson_corr: .2g} p = {pearson_p: .2g}'\
                + '\n' + f'Slope: {slope: .2g} +/- {stderr: 0.2g} p = {p_val: .2g}'\
                + '\n' f'R2: {r2: .2g}'

            f1 = plt.figure()
            plt.plot(ad_measure_array, pred_error, 'ro', label=label)
            plt.xlabel(ad_measure)
            plt.ylabel('Squared Error')
            plt.title(algorithm)
            plt.legend(loc="upper left")

            file_name = f'{ad_measure}_test_set{i}.png'
            file_path = os.path.join(out_path, algorithm, file_name)
            plt.tight_layout()
            plt.savefig(file_path)

            return None
