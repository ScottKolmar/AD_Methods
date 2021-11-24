### IMPORTS ###
import pandas as pd
import numpy as np
import os
import pprint as pp
import matplotlib.pyplot as plt

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

class DataSet():

    def __init__(self, csv, size):
        self.csv = csv
        self.dataset_name = csv.split('\\')[-1].split('_')[0]
        self.df = pd.read_csv(csv, header=0, index_col=0).sample(size, random_state=42)
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

        self.score_dicts = {}

        self.ad_measures = {
            'average_cosine_similarities': [],
            'mahalanobis_distances': []
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
            neighbors.fit(X=self.X_train[i])
            dists, indices = neighbors.kneighbors(X=self.X_test_np[i], n_neighbors=5, return_distance=True)
            self.indices.append(indices)

        return None

    def calc_average_cosine_similarity(self):
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

    def calc_kNN_error_corr(self, n_neighbors, **kwargs):
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
    
    def calc_random_forest_error_corr(self, **kwargs):
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
                'mahalanobis_distances': [],
                'pred_stderr': []
            }
        }

        # Loop through each training set
        for i in range(len(self.X_train)):

            # Fit random forest and create tree dictionary
            rf = RandomForestRegressor(**kwargs)
            rf.fit(self.X_train[i], self.y_train[i])
            self.score_dicts['RF']['Algorithm']['Estimators'].append(rf.get_params())
            self.score_dicts['RF']['Algorithm']['tree_preds'].append({})

            # Predit with each estimator (tree)
            for tree in range(rf.n_estimators):
                self.score_dicts['RF']['Algorithm']['tree_preds'][i][f'tree_{tree}'] = rf.estimators_[tree].predict(self.X_test[i])

            # Predict with ensemble
            y_pred = rf.predict(self.X_test[i])
            self.score_dicts['RF']['Algorithm']['Predictions'].append(y_pred)

            # Calculate squared error for each prediction
            squared_error = (self.y_test[i] - y_pred)**2
            self.score_dicts['RF']['Algorithm']['Squared_error'].append(squared_error)

        return None
    
    def calculate_tree_stderr(self):
        """ Calculates the standard error for a single test set entry from the ensemble of single
        tree predictions in a random forest estimator."""

        # Loop through each set of results
        for i in range(len(self.X_train)):
            
            # Calculate standard error for each test set entry mean prediction using tree prediction arrays
            tree_df = pd.DataFrame.from_dict(data=self.score_dicts['RF']['tree_preds'][i], orient='columns')
            pred_stderr = tree_df.std(axis=1)
            self.score_dicts['RF']['Results']['pred_stderr'].append(pred_stderr)

        return None
    
    def calculate_stats(self, algorithm, ad_measure):
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
            