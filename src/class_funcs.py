### IMPORTS ###
import pandas as pd
import numpy as np
import os
import pprint as pp
import matplotlib.pyplot as plt
from scipy import spatial
from scipy.spatial import distance

# Math imports
from scipy.stats import pearsonr, linregress
from scipy.spatial.distance import cosine

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score

# Sklearn algorithms
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessRegressor

# Sklearn boundaries
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

### FUNCTIONS ###

####################
# DISTANCE FUNCTIONS
####################

def euclidean(v1, v2):
    distance = np.sqrt(np.sum((v1-v2)**2))
    return distance

def cityblock(v1, v2):
    distance = np.sum(abs(v1-v2))
    return distance

def minkowski(v1, v2, n=2):
    distance = np.sum(abs(v1-v2)**n)**(1/n)
    return distance

def chebyshev(v1, v2):
    distance = np.max(abs(v1-v2))
    return distance

def sorensen(v1, v2):
    A = np.sum(abs(v1-v2))
    d = len(v1)
    distance = A/np.sum(v1+v2)
    return distance

def gower(v1, v2):
    A = np.sum(abs(v1-v2))
    d = len(v1)
    distance = A/d
    return distance

def sorgel(v1, v2):
    A = np.sum(abs(v1-v2))
    d = len(v1)
    distance = A/np.sum(np.maximum(v1,v2))
    return distance

def kulczynski(v1, v2):
    A = np.sum(abs(v1-v2))
    d = len(v1)
    distance = A/np.sum(np.minimum(v1,v2))
    return distance

def canberra(v1,v2):
    distance = spatial.distance.canberra(v1,v2)
    return distance

def lorentzian(v1,v2):
    A = np.sum(abs(v1-v2))
    d = len(v1)
    distance = np.sum(np.log(1+abs(v1-v2)))
    return distance

def czekanowski(v1,v2):
    A = np.sum(abs(v1-v2))
    distance = A/np.sum(v1+v2)
    return distance

def ruzicka(v1,v2):
    distance = 1-np.sum(np.minimum(v1,v2))/np.sum(np.maximum(v1,v2))
    return distance

def tanimoto(v1,v2):
    distance = np.sum(np.maximum(v1,v2)-np.minimum(v1,v2))/np.sum(np.maximum(v1,v2))
    return distance

def cosine_distance(v1,v2):
    distance = cosine(v1,v2)
    return distance

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
        
        self.indices = {
            'X_test':{
                'cosine_distance': [],
                'euclidean': [],
                'cityblock': [],
                'minkowski': [],
                'chebyshev': [],
                'sorensen': [],
                'gower': [],
                'sorgel': [],
                'kulczynski': [],
                'canberra': [],
                'lorentzian': [],
                'czekanowski': [],
                'ruzicka': [],
                'tanimoto': []
            },
            'X_train': {
                'cosine_distance': [],
                'euclidean': [],
                'cityblock': [],
                'minkowski': [],
                'chebyshev': [],
                'sorensen': [],
                'gower': [],
                'sorgel': [],
                'kulczynski': [],
                'canberra': [],
                'lorentzian': [],
                'czekanowski': [],
                'ruzicka': [],
                'tanimoto': []
            }
        }

        self.distances = {
            'X_test': {
                'cosine_distance': [],
                'euclidean': [],
                'cityblock': [],
                'minkowski': [],
                'chebyshev': [],
                'sorensen': [],
                'gower': [],
                'sorgel': [],
                'kulczynski': [],
                'canberra': [],
                'lorentzian': [],
                'czekanowski': [],
                'ruzicka': [],
                'tanimoto': []
            },
            'X_train': {
                'cosine_distance': [],
                'euclidean': [],
                'cityblock': [],
                'minkowski': [],
                'chebyshev': [],
                'sorensen': [],
                'gower': [],
                'sorgel': [],
                'kulczynski': [],
                'canberra': [],
                'lorentzian': [],
                'czekanowski': [],
                'ruzicka': [],
                'tanimoto': [],
            }
        }

        self.mean_training_set_distances = {
                'cosine_distance': [],
                'euclidean': [],
                'cityblock': [],
                'minkowski': [],
                'chebyshev': [],
                'sorensen': [],
                'gower': [],
                'sorgel': [],
                'kulczynski': [],
                'canberra': [],
                'lorentzian': [],
                'czekanowski': [],
                'ruzicka': [],
                'tanimoto': [],
            }

        self.score_dicts = {}

        self.n_neighbors = None

        self.test_set_ad_measures = {
            'distance': {
                'average_cosine_distance': [],
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
            },
            'boundary': {
                'one_class_svm': [],
                'robust_covariance': [],
                'isolation_forest': [],
                'local_outlier_factor': []
            }
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

    def get_test_and_training_set_neighbors(self, n_neighbors, **kwargs):
        """ For every distance function,
            calculates the n nearest neighbors for every entry in the training and test sets,
            calculates the indices and distances for each neighbor,
            calculates the average distance from the n neighbors, and
            calculates the average n-neighbor distance over the entire training set.
        """

        # Store neighbors as class variable
        self.n_neighbors = n_neighbors

        # Loop through every distance function
        for function in [euclidean, cityblock, minkowski, chebyshev, sorensen, gower, sorgel, kulczynski, canberra, lorentzian, czekanowski, ruzicka, tanimoto, cosine_distance]:
            
            # Loop through each training/test set
            for i in range(len(self.X_train)):
                neighbors = NearestNeighbors(n_neighbors=self.n_neighbors, metric = function, **kwargs)
                neighbors.fit(X=self.X_train_np[i])

                # Calculate for test set
                dists, indices = neighbors.kneighbors(X=self.X_test_np[i], n_neighbors=self.n_neighbors, return_distance=True)
                self.indices['X_test'][function.__name__].append(indices)
                self.distances['X_test'][function.__name__].append(dists)
                row_mean_dists = np.mean(dists, axis=1)
                self.test_set_ad_measures['distance'][f'average_{function.__name__}'].append(row_mean_dists)

                # Calculate for training test
                training_dists, training_indices = neighbors.kneighbors(X=self.X_train_np[i], n_neighbors = self.n_neighbors, return_distance=True)
                self.indices['X_train'][function.__name__].append(training_indices)
                self.distances['X_train'][function.__name__].append(training_dists)
                row_mean_training_dists = np.mean(training_dists, axis=1)
                self.mean_training_set_distances[function.__name__].append(np.mean(row_mean_training_dists))

        return None

##############################
# AD MEASURES
##############################

    def calculate_boundary_one_class_svm(self, nu=0.5, **kwargs):
        """ Calculates distance to a boundary using one class svm."""

        for i in range(len(self.X_test)):

            one_class = OneClassSVM(nu=nu, **kwargs)
            one_class.fit(self.X_train_np[i])
            distances_to_boundary = one_class.decision_function(self.X_test_np[i])
            self.test_set_ad_measures['boundary']['one_class_svm'].append(distances_to_boundary)

        return None

    def calculate_boundary_robust_covariance(self, **kwargs):
        """ Calculates distance to a boundary using robust covariance."""

        for i in range(len(self.X_test)):

            rob_cov = EllipticEnvelope(**kwargs)
            rob_cov.fit(self.X_train_np[i])
            distances_to_boundary = rob_cov.decision_function(self.X_test_np[i])
            self.test_set_ad_measures['boundary']['robust_covariance'].append(distances_to_boundary)

        return None

    def calculate_boundary_isolation_forest(self, **kwargs):
        """ Calculates distance to a boundary using isolation forest."""

        for i in range(len(self.X_test)):
            
            forest = IsolationForest(**kwargs)
            forest.fit(self.X_train_np[i])
            distances_to_boundary = forest.decision_function(self.X_test_np[i])
            self.test_set_ad_measures['boundary']['isolation_forest'].append(distances_to_boundary)

        return None

    def calculate_boundary_local_outlier_factor(self, **kwargs):
        """ Calculates distance to a boundary using local outlier factor."""

        for i in range(len(self.X_test)):

            lof = LocalOutlierFactor(novelty = True, **kwargs)
            lof.fit(self.X_train_np[i])
            distances_to_boundary = lof.decision_function(self.X_test_np[i])
            self.test_set_ad_measures['boundary']['local_outlier_factor'].append(distances_to_boundary)

        return None

#########################
# Algorithm predictions
#########################

    def predict_kNN(self, n_neighbors, **kwargs):
        """ Generates predictions using kNN and calculates the prediction error."""

        # Define class information
        self.score_dicts['kNN'] = {
            'Algorithm': {
                'Estimators': [],
                'Predictions': [],
                'Squared_error': []
                },
            'Results':{
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            },
            'Ratios': {
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            },
            'Differences': {
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            }
        }

        # Loop through each training set
        for i in range(len(self.X_train)):

            # Fit knn and add params to class object
            knn = KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs)
            knn.fit(self.X_train[i], self.y_train[i])
            self.score_dicts['kNN']['Algorithm']['Estimators'].append(knn.get_params())

            # Make and store predictions
            y_pred = knn.predict(self.X_test[i])
            self.score_dicts['kNN']['Algorithm']['Predictions'].append(y_pred)

            # Calculate and store prediction error
            squared_error = (self.y_test[i] - y_pred)**2
            self.score_dicts['kNN']['Algorithm']['Squared_error'].append(squared_error)
        
        return None
    
    def predict_random_forest(self, **kwargs):
        """ Generates predictions using Random Forest, calculates the prediction error, and calculates
        the Pearson Correlation Coefficient between the prediction error and the AD measures."""

        self.score_dicts['RF'] = {
            'Algorithm': {
                'Estimators': [],
                'Predictions': [],
                'Squared_error': [],
                'Tree_predictions': []
            },
            'Intrinsic': {
                'arrays': [],
                'description': 'Standard deviations of the predictions of each tree in the random forest.'
            },
            'Results': {
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            },
            'Ratios': {
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            },
            'Differences': {
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            }
        }

        self.score_dicts['RF']['Results']['Intrinsic'] = []
        

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
    
    def predict_svr(self, **kwargs):
        """ Generates predictions using support vector machines and calculates the prediction error."""

        # Define class information
        self.score_dicts['SVR'] = {
            'Algorithm': {
                'Estimators': [],
                'Predictions': [],
                'Squared_error': []
                },
            'Results':{
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            },
            'Ratios': {
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            },
            'Differences': {
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            }
        }

        # Loop through each training set
        for i in range(len(self.X_train)):

            # Fit knn and add params to class object
            svr = SVR(**kwargs)
            svr.fit(self.X_train[i], self.y_train[i])
            self.score_dicts['SVR']['Algorithm']['Estimators'].append(svr.get_params())

            # Make and store predictions
            y_pred = svr.predict(self.X_test[i])
            self.score_dicts['SVR']['Algorithm']['Predictions'].append(y_pred)

            # Calculate and store prediction error
            squared_error = (self.y_test[i] - y_pred)**2
            self.score_dicts['SVR']['Algorithm']['Squared_error'].append(squared_error)
        
        return None

    def predict_gradient_boosted_trees(self, **kwargs):
        """ Generates predictions using gradient boosted trees and calculates prediction error."""

        # Define class information
        self.score_dicts['GBT'] = {
            'Algorithm': {
                'Estimators': [],
                'Predictions': [],
                'Tree_predictions': [],
                'Squared_error': []
                },
            'Intrinsic': {
                'arrays': [],
                'description': 'Standard deviations of the predictions of each tree in the boosted random forest.'
            },
            'Results':{
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            },
            'Ratios': {
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            },
            'Differences': {
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            }
        }

        self.score_dicts['GBT']['Results']['Intrinsic'] = []

        # Loop through each training set
        for i in range(len(self.X_train)):

            # Fit knn and add params to class object
            gbt = GradientBoostingRegressor(**kwargs)
            gbt.fit(self.X_train[i], self.y_train[i])
            self.score_dicts['GBT']['Algorithm']['Estimators'].append(gbt.get_params())
            self.score_dicts['GBT']['Algorithm']['Tree_predictions'].append({})

            # Make and store predictions
            y_pred = gbt.predict(self.X_test[i])
            self.score_dicts['GBT']['Algorithm']['Predictions'].append(y_pred)

            # Calculate and store prediction error
            squared_error = (self.y_test[i] - y_pred)**2
            self.score_dicts['GBT']['Algorithm']['Squared_error'].append(squared_error)

            # Predict with each estimator (tree)
            for i_tree,tree in enumerate(gbt.estimators_):
                preds = gbt.estimators_[i_tree][0].predict(self.X_test[i])
                self.score_dicts['GBT']['Algorithm']['Tree_predictions'][i][f'tree_{i_tree}'] = preds

        return None

    def predict_gaussian_process(self, **kwargs):
        """ Generates predictions with gaussian process and calculates prediction error."""

        # Define class information
        self.score_dicts['GP'] = {
            'Algorithm': {
                'Estimators': [],
                'Predictions': [],
                'Squared_error': []
                },
            'Intrinsic': {
                'arrays': [],
                'description': 'Standard deviations of prediction for each compound.'
            },
            'Results':{
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            },
            'Ratios': {
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            },
            'Differences': {
                'distance': {k:[] for k in self.ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.ad_measures['boundary'].keys()}
            }
        }

        self.score_dicts['GP']['Results']['Intrinsic'] = []

        # Loop through each training set
        for i in range(len(self.X_train)):

            # Fit knn and add params to class object
            gp = GaussianProcessRegressor(**kwargs)
            gp.fit(self.X_train[i], self.y_train[i])
            self.score_dicts['GP']['Algorithm']['Estimators'].append(gp.get_params())

            # Make and store predictions and stdevs of prediction
            y_pred, y_stds = gp.predict(X=self.X_test[i], return_std=True)
            self.score_dicts['GP']['Algorithm']['Predictions'].append(y_pred)
            self.score_dicts['GP']['Intrinsic']['arrays'].append(y_stds)

            # Calculate and store prediction error
            squared_error = (self.y_test[i] - y_pred)**2
            self.score_dicts['GP']['Algorithm']['Squared_error'].append(squared_error)

        return None

#####################
# GET RESULTS
#####################
    
    def calculate_tree_std(self, algorithm):
        """ Calculates the standard deviation for a single test set entry from the ensemble of single
        tree predictions in a random forest estimator."""

        # Loop through each set of results
        for i in range(len(self.X_train)):
            
            # Calculate standard error for each test set entry mean prediction using tree prediction arrays
            tree_df = pd.DataFrame.from_dict(data=self.score_dicts[algorithm]['Algorithm']['Tree_predictions'][i], orient='columns')
            pred_std = tree_df.std(axis=1)
            self.score_dicts[algorithm]['Intrinsic']['arrays'].append(pred_std)

        return None
    
    def calculate_global_stats(self, algorithm, measure_type, ad_measure, print_res):
        """ Calculates pearson correlation coefficient, slope, and R2 for the
        provided distance or boundary based applicability domain measure, for the provided algorithm.
        
        algorithm: Set to 'RF', 'kNN', 'GP', 'GBT', or 'SVR' according to desired algorithm
        measure_type: Set to 'distance' or 'boundary' according to the desired ad measure type
        ad_measure: Set to any of dataset.ad_measures.keys() according to desired specific ad measure
        print_res (bool): Set to True to print the results as they are calculated

        """

        # Set algorithm dictionary
        alg_dict = self.score_dicts[algorithm]

        # Loop through each set of results
        for i in range(len(self.X_train)):

            # Set dependent variable
            pred_error = alg_dict['Algorithm']['Squared_error'][i]

            # Set empty result dictionary if there is no data for a measure
            if not self.ad_measures[measure_type][ad_measure]:
                result_dict = {}
            
            else:
                # Set array if data exists
                ad_measure_array = self.ad_measures[measure_type][ad_measure][i]

                # Calculate pearson r
                try:
                    corr, corr_p = pearsonr(ad_measure_array, pred_error)
                except ValueError:
                    corr = np.nan
                    corr_p = np.nan

                # Calculate linear fit
                try:
                    lin_result = linregress(ad_measure_array, pred_error)
                except ValueError:
                    lin_result = np.nan

                # Store results
                result_dict = {
                    'Pearson_correlation': corr,
                    'Pearson_p_val': corr_p,
                    'Slope': lin_result.slope,
                    'Stderr': lin_result.stderr,
                    'p_val': lin_result.pvalue,
                    'R2': lin_result.rvalue**2
                }
                alg_dict['Results'][measure_type][ad_measure].append(result_dict)

                # Print results
                if print_res:
                    printer = pp.PrettyPrinter(indent=4)
                    printer.pprint(alg_dict['Results'][measure_type][ad_measure][i])

        return None
    
    def calculate_intrinsic_stats(self, algorithm, print_res):
        """ Calculates the pearson correlation coefficient, slope, and R2 for the
        provided algorithm-intrinsic applicability domain measure, for the provided algorithm."""

        # Set algorithm dictionary
        alg_dict = self.score_dicts[algorithm]

        # Loop through each set of results
        for i in range(len(self.X_train)):

            # Set dependent variable
            pred_error = alg_dict['Algorithm']['Squared_error'][i]

            # Set empty result dictionary if there is no data for a measure
            if not alg_dict['Intrinsic']['arrays']:
                result_dict = {}
            
            else:
                # Set array if data exists
                ad_measure_array = alg_dict['Intrinsic']['arrays'][i]

                # Calculate pearson r
                try:
                    corr, corr_p = pearsonr(ad_measure_array, pred_error)
                except ValueError:
                    corr = np.nan
                    corr_p = np.nan

                # Calculate linear fit
                try:
                    lin_result = linregress(ad_measure_array, pred_error)
                except ValueError:
                    lin_result = np.nan

                # Store results
                result_dict = {
                    'Pearson_correlation': corr,
                    'Pearson_p_val': corr_p,
                    'Slope': lin_result.slope,
                    'Stderr': lin_result.stderr,
                    'p_val': lin_result.pvalue,
                    'R2': lin_result.rvalue**2
                }
                alg_dict['Results']['Intrinsic'].append(result_dict)

                # Print results
                if print_res:
                    printer = pp.PrettyPrinter(indent=4)
                    printer.pprint(alg_dict['Results']['Intrinsic'][i])

        return None

    def calculate_ratios(self, algorithm, measure_type, ad_measure, print_res):
        """ Calculates ratios of RMSEinsideAD/RMSEoutsideAD and R2insideAD/R2outsideAD for a range of
        thresholds."""

        # Loop through each test set
        for i in range(len(self.X_test)):

            # Calculate thresholds for AD
            if not self.ad_measures[measure_type][ad_measure]:
                return f'No data for {measure_type}: {ad_measure}'
            
            ad_measure_array = self.ad_measures[measure_type][ad_measure][i]
            thresholds = [np.percentile(ad_measure_array, x) for x in range(5,100, 5)]
            
            # Make empty ratio lists
            rmse_ratios = []
            r2_ratios = []

            # Loop through each threshold
            for thresh in thresholds:

                # Get bool arrays for inside and outside
                bool_inside_AD = (ad_measure_array < thresh)
                bool_outside_AD = (ad_measure_array > thresh)                    

                # Get test set true values for inside and outside
                y_true_inside = self.y_test_np[i][bool_inside_AD]
                y_true_outside = self.y_test_np[i][bool_outside_AD]

                # Get test set predicted values for inside and outside
                y_pred = np.array(self.score_dicts[algorithm]['Algorithm']['Predictions'][i])
                y_pred_inside = y_pred[bool_inside_AD]
                y_pred_outside = y_pred[bool_outside_AD]

                # Set ratios to np.nan if all compounds are inside or outside AD
                if np.sum(bool_inside_AD) == 0 or np.sum(bool_outside_AD) == 0:
                    rmse_ratio = np.nan
                    rmse_ratios.append(rmse_ratio)
                    r2_ratio = np.nan
                    r2_ratios.append(r2_ratio)

                # Proceed to calculation if there are compounds both inside and outside AD
                else:
                    # Get RMSE inside and outside and append to ratio list
                    rmse_inside = mean_squared_error(y_true_inside, y_pred_inside, squared=False)
                    rmse_outside = mean_squared_error(y_true_outside, y_pred_outside, squared=False)
                    rmse_ratio = rmse_inside/rmse_outside
                    rmse_ratios.append(rmse_ratio)

                    # Get R2 inside and outside and append to ratio list
                    r2_inside = r2_score(y_true_inside, y_pred_inside)
                    r2_outside = r2_score(y_true_outside, y_pred_outside)
                    r2_ratio = r2_inside/r2_outside
                    r2_ratios.append(r2_ratio)
            
            # Create ratio dictionary and append to class object
            ratio_dict = {
                'thresholds': thresholds,
                'threshold_percentiles': np.arange(5,100,5),
                'rmse_ratios': rmse_ratios,
                'r2_ratios': r2_ratios
            }
            self.score_dicts[algorithm]['Ratios'][measure_type][ad_measure].append(ratio_dict)

            # Print if parameter set to print
            if print_res:
                pprinter = pp.PrettyPrinter(indent=4)
                pprinter.pprint(ratio_dict)
        
        return None
    
    def calculate_difference(self, algorithm, measure_type, ad_measure, print_res):
        """ Calculates R2inside-R2outside and RMSEoutside-RMSEinside for a range of 
        thresholds for a given algorithm and ad_measure."""

        # Loop through each test set
        for i in range(len(self.X_test)):

            # Calculate thresholds for AD
            if not self.ad_measures[measure_type][ad_measure]:
                return f'No data for {measure_type}: {ad_measure}'
            
            ad_measure_array = self.ad_measures[measure_type][ad_measure][i]
            thresholds = [np.percentile(ad_measure_array, x) for x in range(5,100, 5)]
            
            # Make empty difference lists
            rmse_differences = []
            r2_differences = []

            # Loop through each threshold
            for thresh in thresholds:

                # Get bool arrays for inside and outside
                bool_inside_AD = (ad_measure_array < thresh)
                bool_outside_AD = (ad_measure_array > thresh)                    

                # Get test set true values for inside and outside
                y_true_inside = self.y_test_np[i][bool_inside_AD]
                y_true_outside = self.y_test_np[i][bool_outside_AD]

                # Get test set predicted values for inside and outside
                y_pred = np.array(self.score_dicts[algorithm]['Algorithm']['Predictions'][i])
                y_pred_inside = y_pred[bool_inside_AD]
                y_pred_outside = y_pred[bool_outside_AD]

                # Set ratios to np.nan if all compounds are inside or outside AD
                if np.sum(bool_inside_AD) == 0 or np.sum(bool_outside_AD) == 0:
                    rmse_difference = np.nan
                    rmse_differences.append(rmse_difference)
                    r2_difference = np.nan
                    r2_differences.append(r2_difference)

                # Proceed to calculation if there are compounds both inside and outside AD
                else:
                    # Get RMSE inside and outside and append to ratio list
                    rmse_inside = mean_squared_error(y_true_inside, y_pred_inside, squared=False)
                    rmse_outside = mean_squared_error(y_true_outside, y_pred_outside, squared=False)
                    rmse_difference = rmse_outside - rmse_inside
                    rmse_differences.append(rmse_difference)

                    # Get R2 inside and outside and append to ratio list
                    r2_inside = r2_score(y_true_inside, y_pred_inside)
                    r2_outside = r2_score(y_true_outside, y_pred_outside)
                    r2_difference = r2_inside - r2_outside
                    r2_differences.append(r2_difference)
            
            # Create ratio dictionary and append to class object
            difference_dict = {
                'thresholds': thresholds,
                'threshold_percentiles': np.arange(5,100,5),
                'rmse_differences': rmse_differences,
                'r2_differences': r2_differences
            }
            self.score_dicts[algorithm]['Differences'][measure_type][ad_measure].append(difference_dict)

            # Print if parameter set to print
            if print_res:
                pprinter = pp.PrettyPrinter(indent=4)
                pprinter.pprint(difference_dict)
        
        return None

    def plot_comparisons(self, comparison, algorithm, measure_type, ad_measure, out_path):
        """ Plots ratios of RMSEinsideAD/RMSEoutsideAD and R2insideAD/R2outsideAD for a range of
        thresholds for a given algorithm and ad_measure.
        
        comparison: Set to 'Ratios' to plot ratio data and 'Differences' to plot difference data
        algorithm: Set to 'RF', 'kNN', 'GP', 'GBT', or 'SVR' for desired algorithm
        measure_type: Set to 'distance' or 'boundary according to desired ad measure type
        ad_measure: Set to any of dataset.ad_measures.measure_type.keys() according to desired ad measure
        out_path: Path for parent PNG folder. Algorithm name is append to the path automatically.
        """

        # Loop through all test sets
        for i in range(len(self.X_test)):
            
            # Get the data to be plotted
            if comparison == 'Ratios':
                if not self.score_dicts[algorithm]['Ratios'][measure_type][ad_measure][i]:
                    return f'No ratio data for {measure_type}: {ad_measure}'
                comparison_dict = self.score_dicts[algorithm]['Ratios'][measure_type][ad_measure][i]
                rmse_data = comparison_dict['rmse_ratios']
                r2_data = comparison_dict['r2_ratios']
                rmse_label = 'RMSEin/RMSEout'
                r2_label = 'R2in/R2out'

            elif comparison == 'Differences':
                if not self.score_dicts[algorithm]['Differences'][measure_type][ad_measure]:
                    return f'No difference data for {measure_type}: {ad_measure}'
                comparison_dict = self.score_dicts[algorithm]['Differences'][measure_type][ad_measure][i]
                rmse_data = comparison_dict['rmse_differences']
                r2_data = comparison_dict['r2_differences']
                rmse_label = 'RMSEout-RMSEin'
                r2_label = 'R2in-R2out'

            # If no data, exit function
            if not r2_data:
                return f'No data for {measure_type}: {ad_measure}'

            # Set x axis for plot
            thresh_perc = comparison_dict['threshold_percentiles']

            # Make figure with two different y-axes on same plot
            fig, ax1 = plt.subplots()

            # Plot R2 data
            color = 'tab:red'
            ax1.set_xlabel('Threshold Percentile')
            ax1.set_ylabel(r2_label, color=color)
            ax1.plot(thresh_perc, r2_data, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            # Set duplicate x axis
            ax2 = ax1.twinx()

            # Plot RMSE data
            color = 'tab:blue'
            ax2.set_ylabel(rmse_label, color=color)
            ax2.plot(thresh_perc, rmse_data, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            # Set figure title and layout
            fig.suptitle(f'{algorithm}, {ad_measure}')
            fig.tight_layout()

            # Save figure
            file_name = f'{comparison}_{ad_measure}_test_set{i}.png'
            file_path = os.path.join(out_path, algorithm, file_name)
            plt.savefig(file_path)

        return None

    def plot_correlations(self, algorithm, measure_type, ad_measure, out_path):
        """ Plots correlations between prediction errors and AD measures.

        algorithm: Set to 'RF', 'kNN', 'GP', 'GBT', or 'SVR' for desired algorithm
        measure_type: Set to 'distance' or 'boundary according to desired ad measure type
        ad_measure: Set to any of dataset.ad_measures.measure_type.keys() according to desired ad measure
        out_path: Path for parent PNG folder. Algorithm name is append to the path automatically.
        
        """

        # Loop through all results
        for i in range(len(self.X_test)):

            # Get ad measure array
            if ad_measure == 'Intrinsic':
                ad_measure_array = self.score_dicts[algorithm][ad_measure]['arrays'][i]
            else:
                # Skip measure if empty array
                if not self.ad_measures[measure_type][ad_measure]:
                    print(f'No data for {measure_type}: {ad_measure}.')

                # Get array if it exists
                else:
                    ad_measure_array = self.ad_measures[measure_type][ad_measure][i]
            
                    # Get prediction error array
                    pred_error = self.score_dicts[algorithm]['Algorithm']['Squared_error'][i]

                    # Get result stats
                    result_dict = self.score_dicts[algorithm]['Results'][measure_type][ad_measure][i]
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
