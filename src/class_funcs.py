### IMPORTS ###
import pandas as pd
import numpy as np
import os
import pprint as pp
import matplotlib.pyplot as plt
import pickle
from scipy import spatial
from scipy.spatial import distance

# Math imports
from scipy.stats import pearsonr, linregress
from scipy.spatial.distance import cosine

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
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

function_list = ['cosine_distance',
                'euclidean',
                'cityblock',
                'minkowski',
                'chebyshev',
                'sorensen',
                'gower',
                'sorgel',
                'kulczynski',
                'canberra',
                'lorentzian',
                'czekanowski',
                'ruzicka',
                'tanimoto']

boundary_list = ['one_class_svm',
                'robust_covariance',
                'isolation_forest',
                'local_outlier_factor']

###################
# UTILS
##################

def drop_infs(df):
    """
    Drops columns which contain infinite values in a dataframe.

    Parameters:
    df (dataframe): Dataframe to drop infinite values. (Default = 600)

    Returns:
        df (dataframe): Dataframe with infinite values dropped.
    """
    cols = [x for x in df.columns if df[x].dtype == 'object']
    df = df.drop(columns=cols)
    df[df.columns] = df[df.columns].astype(float)
    df = df.fillna(0.0).astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how='any')
    return df

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

#####################
# Algorithm CLASS
#####################

class Algorithm():

    def __init__(self, estimator, recursive_features):

        if recursive_features:
            rfe = RFE(estimator, n_features_to_select=recursive_features)
            self.estimator = rfe
        else:
            self.estimator = estimator
        self.name = self.estimator.__class__.__name__

        self.predictions = []
        self.squared_error = []
        self.absolute_error = []
        self.recursive_features = recursive_features

        self.Results = {
            'squared_error':{
                'distance':{k:[] for k in function_list},
                'boundary': {k:[] for k in boundary_list}
            },
            'absolute_error':{
                'distance':{k:[] for k in function_list},
                'boundary': {k:[] for k in boundary_list}
            }   
        }

        self.Ratios = {
            'distance':{k:[] for k in function_list},
            'boundary': {k:[] for k in boundary_list}
            }
            
        self.Differences = {
            'distance':{k:[] for k in function_list},
            'boundary': {k:[] for k in boundary_list}
            }

######################
# DATASET FUNCTIONS
######################

class DataSet():

    def __init__(self, csv, size, **kwargs):
        self.csv = csv
        self.dataset_name = csv.split('\\')[-1].split('_')[0]
        self.df = pd.read_csv(csv, header=0, index_col=0)
        self.df = drop_infs(self.df)
        if size:
            self.df = self.df.sample(size, **kwargs)
        self.X = self.df.iloc[:,:-1]
        self.y = self.df.iloc[:,-1]

        self.var_filter_ = {'filtered': False, 'Value': None}
        self.corr_filter_ = {'filtered': False, 'Value': None}
        self.scaled_ = False

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
            'boundary': {
                'one_class_svm': [],
                'robust_covariance': [],
                'isolation_forest': [],
                'local_outlier_factor': []
            }
        }
    
    def normalize_data(self):
        """ Normalizes data."""

        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(self.X)
        scaled_df = pd.DataFrame(scaled_X, index = self.X.index, columns = self.X.columns)
        self.X = scaled_df
        self.scaled_ = True

        return None

    def apply_variance_filter(self, threshold_percentile):
        """ Filters the features in the dataset on a percentile of the total variance. Every feature
        below the variance threshold is dropped.

        threshold_percentile: Percentile value for the filter (i.e. 50 for 50%
        )
        """
        # Get absolute threshold value from percentile
        threshold = np.percentile(self.X.var(), threshold_percentile)

        # Get variance and assign to dataframe where feature variables are rows
        variance = self.X.var()
        df_var = pd.DataFrame(data = {'variance': variance}, index = self.X.columns)

        # Drop the low variance rows
        df_low_v_dropped = df_var[df_var['variance'] > threshold]

        # Filter the dataset's X dataframe by the selected feature variables
        self.X = self.X[df_low_v_dropped.index]

        # Assign dataset variables
        self.var_filter_ = {'filtered': True, 'Value': threshold}

        return None
    
    def apply_correlation_filter(self, threshold):
        """ Filters the feature in the dataset on an absolute correlation threshold. Any features which correlate
        to one another above the provided threshold are dropped.
        
        threshold: Absolute correlation threshold (i.e. 0.95)

        """

        # Create correlation matrix
        corr_matrix = self.X.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        # Drop features
        self.X.drop(to_drop, axis=1, inplace=True)

        # Reassign dataset variables
        self.num_features = len(self.X.columns)
        self.features = self.X.columns
        self.corr_filter_ = {'filtered': True, 'Value': threshold}

        return None

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
                self.test_set_ad_measures['distance'][function.__name__].append(row_mean_dists)

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
    def predict_algorithm(self, algorithm):
        """
        Makes predictions with an Algorithm class object and stores appropriate class variables.

        algorithm: Sklearn regressor estimator object, such as SVR(), RandomForestRegressor(), etc.
        recursive_features (int): Number of recursive features to use; if None, no recursive feature elimination
        is used

        """

        # Loop through each training set
        for i in range(len(self.X_train)):

            # Fit kNN and predict
            algorithm.estimator.fit(self.X_train[i], self.y_train[i])
            y_pred = algorithm.estimator.predict(self.X_test[i])
            
            # Store predictions
            algorithm.predictions.append(y_pred)

            # Calculate and store prediction error
            squared_error = (self.y_test[i] - y_pred)**2
            algorithm.squared_error.append(squared_error)
            absolute_error = abs(self.y_test[i] - y_pred)
            algorithm.absolute_error.append(absolute_error)
        
        return None

    def predict_kNN(self, n_neighbors, recursive_features, **kwargs):
        """ Generates predictions using kNN and calculates the prediction error.

        n_neighbors (int): Number of neighbors used in prediction
        recursive_features (int): If an integer is provided, recursive feature elimination will be applied before
                         prediction, using the provided number of features
        """

        # Define class information
        self.score_dicts['kNN'] = {
            'Algorithm': {
                'Estimators': [],
                'Predictions': [],
                'Squared_error': [],
                'Absolute_error': [],
                'Recursive_features': recursive_features
                },
            'Results':{
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
            },
            'Ratios': {
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
            },
            'Differences': {
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
            }
        }

        # Loop through each training set
        for i in range(len(self.X_train)):

            # Instantiate kNN
            knn = KNeighborsRegressor(n_neighbors=n_neighbors, **kwargs)

            # Use RFE if recursive features enabled
            if recursive_features:
                # Fit RFE with underlying kNN and predict
                rfe = RFE(knn, n_features_to_select=recursive_features)
                rfe.fit(self.X_train[i], self.y_train[i])
                self.score_dicts['kNN']['Algorithm']['Estimators'].append(rfe.get_params())
                y_pred = rfe.predict(self.X_test[i])
            else:
                # Fit kNN and predict
                knn.fit(self.X_train[i], self.y_train[i])
                self.score_dicts['kNN']['Algorithm']['Estimators'].append(knn.get_params())
                y_pred = knn.predict(self.X_test[i])
            
            # Store predictions
            self.score_dicts['kNN']['Algorithm']['Predictions'].append(y_pred)

            # Calculate and store prediction error
            squared_error = (self.y_test[i] - y_pred)**2
            self.score_dicts['kNN']['Algorithm']['Squared_error'].append(squared_error)
            absolute_error = abs(self.y_test[i] - y_pred)
            self.score_dicts['kNN']['Algorithm']['Absolute_error'].append(absolute_error)
        
        return None
    
    def predict_random_forest(self, **kwargs):
        """ Generates predictions using Random Forest, calculates the prediction error, and calculates
        the Pearson Correlation Coefficient between the prediction error and the AD measures."""

        self.score_dicts['RF'] = {
            'Algorithm': {
                'Estimators': [],
                'Predictions': [],
                'Squared_error': [],
                'Absolute_error': [],
                'Tree_predictions': []
            },
            'Intrinsic': {
                'arrays': [],
                'description': 'Standard deviations of the predictions of each tree in the random forest.'
            },
            'Results': {
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
            },
            'Ratios': {
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
            },
            'Differences': {
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
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
            absolute_error = abs(self.y_test[i] - y_pred)
            self.score_dicts['RF']['Algorithm']['Absolute_error'].append(absolute_error)

        return None
    
    def predict_svr(self, **kwargs):
        """ Generates predictions using support vector machines and calculates the prediction error."""

        # Define class information
        self.score_dicts['SVR'] = {
            'Algorithm': {
                'Estimators': [],
                'Predictions': [],
                'Squared_error': [],
                'Absolute_error': []
                },
            'Results':{
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
            },
            'Ratios': {
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
            },
            'Differences': {
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
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
            absolute_error = abs(self.y_test[i] - y_pred)
            self.score_dicts['SVR']['Algorithm']['Absolute_error'].append(absolute_error)
        
        return None

    def predict_gradient_boosted_trees(self, **kwargs):
        """ Generates predictions using gradient boosted trees and calculates prediction error."""

        # Define class information
        self.score_dicts['GBT'] = {
            'Algorithm': {
                'Estimators': [],
                'Predictions': [],
                'Tree_predictions': [],
                'Squared_error': [],
                'Absolute_error': []
                },
            'Intrinsic': {
                'arrays': [],
                'description': 'Standard deviations of the predictions of each tree in the boosted random forest.'
            },
            'Results':{
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
            },
            'Ratios': {
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
            },
            'Differences': {
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
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
            absolute_error = abs(self.y_test[i] - y_pred)
            self.score_dicts['GBT']['Algorithm']['Absolute_error'].append(absolute_error)

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
                'Squared_error': [],
                'Absolute_error': []
                },
            'Intrinsic': {
                'arrays': [],
                'description': 'Standard deviations of prediction for each compound.'
            },
            'Results':{
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
            },
            'Ratios': {
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
            },
            'Differences': {
                'distance': {k:[] for k in self.test_set_ad_measures['distance'].keys()},
                'boundary': {k:[] for k in self.test_set_ad_measures['boundary'].keys()}
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
            absolute_error = abs(self.y_test[i] - y_pred)
            self.score_dicts['GP']['Algorithm']['Absolute_error'].append(absolute_error)

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
    
    def calculate_global_stats_class(self, algorithm, print_res):
        """ Calculates correlation coefficient, slope, R2, and p values for each type of prediction error
        and applicability domain measure, for an Algorithm object.

        algorithm: Algorithm object which has stored prediction error results
        print_res (bool): Set to True to print the correlation results
        """

        for i in range(len(self.X_test)):

            for error_metric in ['squared_error', 'absolute_error']:

                pred_error = getattr(algorithm, error_metric)[i]
                
                for ad_measure_type in ['boundary', 'distance']:

                    for key in self.test_set_ad_measures[ad_measure_type].keys():

                        if not self.test_set_ad_measures[ad_measure_type][key]:
                            result_dict = {}
                        else:
                            ad_measure_array = self.test_set_ad_measures[ad_measure_type][key][i]

                            corr, corr_p = pearsonr(ad_measure_array, pred_error)
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
                            getattr(algorithm, 'Results')[error_metric][ad_measure_type][key].append(result_dict)

                        if print_res:
                            printer = pp.PrettyPrinter(indent=4)
                            print(f'METRIC: {error_metric} TYPE: {ad_measure_type} MEASURE: {key}')
                            printer.pprint(result_dict)
                            print('~'*30)
        
        return None
    
    def calculate_global_stats(self, algorithm, measure_type, ad_measure, error_metric, print_res):
        """ Calculates pearson correlation coefficient, slope, and R2 for the
        provided distance or boundary based applicability domain measure, for the provided algorithm.
        
        algorithm: Set to 'RF', 'kNN', 'GP', 'GBT', or 'SVR' according to desired algorithm
        measure_type: Set to 'distance' or 'boundary' according to the desired ad measure type
        ad_measure: Set to any of dataset.ad_measures.keys() according to desired specific ad measure
        error_metric: Error metric to calculate statistics for, 'Squared_error' or 'Absolute_error'
        print_res (bool): Set to True to print the results as they are calculated

        """

        # Set algorithm dictionary
        alg_dict = self.score_dicts[algorithm]

        # Loop through each set of results
        for i in range(len(self.X_train)):

            # Set dependent variable
            pred_error = alg_dict['Algorithm'][error_metric][i]

            # Set empty result dictionary if there is no data for a measure
            if not self.test_set_ad_measures[measure_type][ad_measure]:
                result_dict = {}
            
            else:
                # Set array if data exists
                ad_measure_array = self.test_set_ad_measures[measure_type][ad_measure][i]

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
                    print(f'METRIC: {error_metric} TYPE: {measure_type} MEASURE: {ad_measure}')
                    printer.pprint(alg_dict['Results'][measure_type][ad_measure][i])
                    print('-'*30)

        return None
    
    def calculate_intrinsic_stats(self, algorithm, error_metric, print_res):
        """ Calculates the pearson correlation coefficient, slope, and R2 for the
        provided algorithm-intrinsic applicability domain measure, for the provided algorithm.
        
        algorithm: Algorithm to calculate statistics for, 'kNN', 'RF', 'GP', 'GBT', or 'SVR'
        error_metric: Error metric to calculate statistics for, 'Squared_error' or 'Absolute_error'
        print_res (bool): Set to True to print result dictionary
        """

        # Set algorithm dictionary
        alg_dict = self.score_dicts[algorithm]

        # Loop through each set of results
        for i in range(len(self.X_train)):

            # Set dependent variable
            pred_error = alg_dict['Algorithm'][error_metric][i]

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
    
    def calculate_ratios_class(self, algorithm, print_res):
        """ Calculates RMSE and R2 ratios inside and outside AD for a given algorithm class object.

        algorithm: Algorithm class object with predictions
        print_res (bool): Set to True to print the ratio dictionary

        """

        # Loop through test sets
        for i in range(len(self.X_test)):
            
            # Loop through measure types
            for ad_measure_type in ['distance', 'boundary']:
                
                # Loop through each measure
                for key in self.test_set_ad_measures[ad_measure_type].keys():
                    
                    # Calculate appropriate thresholds
                    if ad_measure_type == 'distance':
                        training_distances = self.distances['X_train'][key][i]
                        training_average_d_neighbors = np.mean(training_distances, axis=1)
                        thresholds = [np.percentile(training_average_d_neighbors, x) for x in range(5,100,5)]
                    elif ad_measure_type == 'boundary':

                        # Catch empty list for a boundary measure
                        if not self.test_set_ad_measures[ad_measure_type][key]:
                            Z = None
                            thresholds = None
                        else:
                            Z = self.test_set_ad_measures[ad_measure_type][key][i]
                            thresholds = np.linspace(Z.min(), 0, 10)
                    
                    # Catch empty array
                    if not self.test_set_ad_measures[ad_measure_type][key]:
                        ad_measure_array = None
                        result_dict = None
                        getattr(algorithm, 'Ratios')[ad_measure_type][key].append(ratio_dict)
                    
                    # Finish loop
                    else:
                        ad_measure_array = self.test_set_ad_measures[ad_measure_type][key][i]

                        # Make empty ratio lists
                        rmse_ratios = []
                        r2_ratios = []

                        # Loop through each threshold
                        for thresh in thresholds:

                            # Get bool arrays for inside and outside
                            if ad_measure_type == 'distance':
                                bool_inside_AD = (ad_measure_array < thresh)
                                bool_outside_AD = (ad_measure_array > thresh)
                            elif ad_measure_type == 'boundary':
                                bool_inside_AD = (ad_measure_array > thresh)
                                bool_outside_AD = (ad_measure_array < thresh)               

                            # Get test set true values for inside and outside
                            y_true_inside = self.y_test_np[i][bool_inside_AD]
                            y_true_outside = self.y_test_np[i][bool_outside_AD]

                            # Get test set predicted values for inside and outside
                            y_pred = np.array(getattr(algorithm, 'predictions')[i])
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
                        if ad_measure_type == 'distance':
                            threshold_percentiles = range(5,100,5)
                        elif ad_measure_type == 'boundary':
                            threshold_percentiles == None

                        ratio_dict = {
                            'thresholds': thresholds,
                            'threshold_percentiles': threshold_percentiles,
                            'rmse_ratios': rmse_ratios,
                            'r2_ratios': r2_ratios
                        }
                        getattr(algorithm, 'Ratios')[ad_measure_type][key].append(ratio_dict)

                        # Print if parameter set to print
                        if print_res:
                            print(f'TYPE: {ad_measure_type} MEASURE: {key}')
                            pprinter = pp.PrettyPrinter(indent=4)
                            pprinter.pprint(ratio_dict)
                            print('*'*30)
        
        return None

    def calculate_ratios(self, algorithm, measure_type, ad_measure, print_res):
        """ Calculates ratios of RMSEinsideAD/RMSEoutsideAD and R2insideAD/R2outsideAD for a range of
        thresholds."""

        # Loop through each test set
        for i in range(len(self.X_test)):

            # Calculate thresholds for AD
            if not self.test_set_ad_measures[measure_type][ad_measure]:
                return f'No data for {measure_type}: {ad_measure}'
            
            if measure_type == 'distance':
                training_set_distances = self.distances['X_train'][ad_measure][i]
                training_set_average_d_neighbors = np.mean(training_set_distances, axis=1)
                thresholds = [np.percentile(training_set_average_d_neighbors, x) for x in range(5,100, 5)]
            
            elif measure_type == 'boundary':
                thresholds = [-1, -0.75, -0.5, 0, 0.5, 0.75, 1]
            
            # Define measure array
            ad_measure_array = self.test_set_ad_measures[measure_type][ad_measure][i]

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
    
    def calculate_differences_class(self, algorithm, print_res):
        """ Calculates differences for R2 and RMSE inside and outside the applicability domain for every
        AD measure.
        
        algorithm: Algorithm class object with predictions
        print_res (bool): Set to True to print difference dictionary
        
        """

        for i in range(len(self.X_test)):

            for ad_measure_type in ['distance', 'boundary']:

                for key in self.test_set_ad_measures[ad_measure_type].keys():
                    
                    if ad_measure_type == 'distance':
                        training_set_distances = self.distances['X_train'][key][i]
                        training_average_d_neighbors = np.mean(training_set_distances, axis=1)
                        thresholds = [np.percentile(training_average_d_neighbors, x) for x in range(5,100, 5)]
                    elif ad_measure_type == 'boundary':

                        # Catch empty list
                        if not self.test_set_ad_measures[ad_measure_type][key]:
                            Z = None
                            thresholds = None
                        else:
                            Z = self.test_set_ad_measures[ad_measure_type][key][i]
                            thresholds = np.linspace(Z.min(), 0, 10)
                    
                    # Catch empty array
                    if not self.test_set_ad_measures[ad_measure_type][key]:
                        ad_measure_array = None
                        result_dict = None
                        getattr(algorithm, 'Differences')[ad_measure_type][key].append(difference_dict)
                    
                    # Finish loop
                    else:
                        ad_measure_array = self.test_set_ad_measures[ad_measure_type][key][i]

                        # Make empty difference lists
                        rmse_differences = []
                        r2_differences = []

                        # Loop through each threshold
                        for thresh in thresholds:

                            # Get bool arrays for inside and outside
                            if ad_measure_type == 'distance':
                                bool_inside_AD = (ad_measure_array < thresh)
                                bool_outside_AD = (ad_measure_array > thresh)
                            elif ad_measure_type == 'boundary':
                                bool_inside_AD = (ad_measure_array > thresh)
                                bool_outside_AD = (ad_measure_array < thresh)                  

                            # Get test set true values for inside and outside
                            y_true_inside = self.y_test_np[i][bool_inside_AD]
                            y_true_outside = self.y_test_np[i][bool_outside_AD]

                            # Get test set predicted values for inside and outside
                            y_pred = np.array(getattr(algorithm, 'predictions')[i])
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
                        if ad_measure_type == 'distance':
                            threshold_percentiles = np.arange(5,100,5)
                        elif ad_measure_type == 'boundary':
                            threshold_percentiles = None
                        difference_dict = {
                            'thresholds': thresholds,
                            'threshold_percentiles': threshold_percentiles,
                            'rmse_differences': rmse_differences,
                            'r2_differences': r2_differences
                        }
                        getattr(algorithm, 'Differences')[ad_measure_type][key].append(difference_dict)

                        # Print if parameter set to print
                        if print_res:
                            print(f'TYPE: {ad_measure_type} MEASURE: {key}')
                            pprinter = pp.PrettyPrinter(indent=4)
                            pprinter.pprint(difference_dict)
                            print('+'*30)
        
        return None
    
    def calculate_difference(self, algorithm, measure_type, ad_measure, print_res):
        """ Calculates R2inside-R2outside and RMSEoutside-RMSEinside for a range of 
        thresholds for a given algorithm and ad_measure."""

        # Loop through each test set
        for i in range(len(self.X_test)):

            # Calculate thresholds for AD
            if not self.test_set_ad_measures[measure_type][ad_measure]:
                return f'No data for {measure_type}: {ad_measure}'
            
            if measure_type == 'distance':
                training_set_distances = self.distances['X_train'][ad_measure][i]
                training_average_d_neighbors = np.mean(training_set_distances, axis=1)
                thresholds = [np.percentile(training_average_d_neighbors, x) for x in range(5,100, 5)]
            elif measure_type == 'boundary':
                thresholds = [-1, -0.75, -0.5, 0, 0.5, 0.75, 1]

            # Set ad measure array
            ad_measure_array = self.test_set_ad_measures[measure_type][ad_measure][i]

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
    
    def plot_comparisons_class(self, algorithm, out_path):
        """ Plots differences and ratios of metrics for compounds inside and outside the applicability domain,
        for each AD measure, for a given algorithm.

        algorithm: Algorithm class object with predictions
        out_path: Absolute path to the results for a dataset
        """

        # Loop through all test sets
        for i in range(len(self.X_test)):

            # Loop through all measure types
            for ad_measure_type in ['distance', 'boundary']:
                
                # Loop through each measure
                for key in self.test_set_ad_measures[ad_measure_type].keys():
                    
                    # Get ratios
                    ratio_dict =  getattr(algorithm, 'Ratios')[ad_measure_type][key][i]
                    rmse_ratio = ratio_dict['rmse_ratios']
                    r2_ratio = ratio_dict['r2_ratios']
                    rmse_ratio_label = 'RMSEin/RMSEout'
                    r2_ratio_label = 'R2in/R2out'

                    # Get differences
                    difference_dict = getattr(algorithm, 'Differences')[ad_measure_type][key][i]
                    rmse_difference = difference_dict['rmse_differences']
                    r2_difference = difference_dict['r2_differences']
                    rmse_difference_label = 'RMSEout-RMSEin'
                    r2_difference_label = 'R2in-R2out'
            

                    # Set xaxis
                    if ad_measure_type == 'distance':
                        ratio_xaxis = ratio_dict['threshold_percentiles']
                        ratio_xlabel = 'Training Set Distance Percentile'
                        difference_xaxis = difference_dict['threshold_percentiles']
                        difference_xlabel = 'Training Set Distance Percentile'
                    elif ad_measure_type == 'boundary':
                        ratio_xaxis = ratio_dict['thresholds']
                        ratio_xlabel = 'Thresholds'
                        difference_xaxis = difference_dict['thresholds']
                        difference_xlabel = 'Thresholds'

                    # Make Ratio figure with two different y-axes on same plot
                    fig, ax1 = plt.subplots()

                    # Plot Ratio R2 data
                    color = 'tab:red'
                    ax1.set_xlabel(ratio_xlabel)
                    ax1.set_ylabel(r2_ratio_label, color=color)
                    ax1.plot(ratio_xaxis, r2_ratio, color=color)
                    ax1.tick_params(axis='y', labelcolor=color)

                    # Set duplicate x axis
                    ax2 = ax1.twinx()

                    # Plot Ratio RMSE data
                    color = 'tab:blue'
                    ax2.set_ylabel(rmse_ratio_label, color=color)
                    ax2.plot(ratio_xaxis, rmse_ratio, color=color)
                    ax2.tick_params(axis='y', labelcolor=color)

                    # Set figure title and layout
                    fig.suptitle(f'{algorithm}, {key}')
                    fig.tight_layout()

                    # Save figure
                    alg_name = algorithm.estimator.__class__.__name__
                    if algorithm.recursive_features:
                        rfe = f'RFE_{algorithm.recursive_features}'
                    else:
                        rfe = 'NoRFE'
                    file_name = f'Ratios_{key}_test_set{i}.png'
                    file_path = os.path.join(out_path, alg_name, rfe,'Ratios', file_name)
                    plt.savefig(file_path)

                    # Make Difference figure
                    fig2, ax3 = plt.subplots()

                    # Plot Difference R2 data
                    color = 'tab:red'
                    ax3.set_xlabel(difference_xlabel)
                    ax3.set_ylabel(r2_difference_label, color=color)
                    ax3.plot(difference_xaxis, r2_difference, color=color)
                    ax3.tick_params(axis='y', labelcolor=color)

                    # Set duplicate x axis
                    ax4 = ax3.twinx()

                    # Plot Ratio RMSE data
                    color = 'tab:blue'
                    ax4.set_ylabel(rmse_difference_label, color=color)
                    ax4.plot(difference_xaxis, rmse_difference, color=color)
                    ax4.tick_params(axis='y', labelcolor=color)

                    # Set figure title and layout
                    fig.suptitle(f'{algorithm}, {key}')
                    fig.tight_layout()

                    # Save figure
                    file_name = f'Differences_{key}_test_set{i}.png'
                    file_path = os.path.join(out_path, algorithm, rfe, 'Differences', file_name)
                    plt.savefig(file_path)

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
                if not self.score_dicts[algorithm]['Ratios'][measure_type][ad_measure]:
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

            # Set xaxis
            if measure_type == 'distance':
                xaxis = comparison_dict['threshold_percentiles']
                xlabel = 'Training Set Distance Percentile'
            elif measure_type == 'boundary':
                xaxis = comparison_dict['thresholds']
                xlabel = 'Thresholds'

            # Make figure with two different y-axes on same plot
            fig, ax1 = plt.subplots()

            # Plot R2 data
            color = 'tab:red'
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(r2_label, color=color)
            ax1.plot(xaxis, r2_data, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            # Set duplicate x axis
            ax2 = ax1.twinx()

            # Plot RMSE data
            color = 'tab:blue'
            ax2.set_ylabel(rmse_label, color=color)
            ax2.plot(xaxis, rmse_data, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            # Set figure title and layout
            fig.suptitle(f'{algorithm}, {ad_measure}')
            fig.tight_layout()

            # Save figure
            file_name = f'{comparison}_{ad_measure}_test_set{i}.png'
            file_path = os.path.join(out_path, algorithm, comparison, file_name)
            plt.savefig(file_path)

        return None
    def plot_correlations_class(self, algorithm, out_path):
        """
        algorithm: Algorithm class object with predictions and correlations
        out_path: Absolute path to PNG folder
        """
        
        # Loop through each test set
        for i in range(len(self.X_test)):
            
            # Loop through each error metric
            for error_metric in ['squared_error', 'absolute_error']:

                # Get prediction error
                pred_error = getattr(algorithm, error_metric)[i]

                # Loop through each measure type
                for ad_measure_type in ['distance', 'boundary']:
                    
                    # Loop through each measure
                    for key in getattr(algorithm, 'Results')[ad_measure_type].keys():
                        
                        # Get ad measure array
                        ad_measure_array = self.test_set_ad_measures[ad_measure_type][key][i]
                        
                        # Get result dictionary
                        result_dict = getattr(algorithm, ad_measure_type)[key][i]
                        pearson_corr = result_dict['Pearson_correlation']
                        pearson_p = result_dict['Pearson_p_val']
                        slope = result_dict['Slope']
                        stderr = result_dict['Stderr']
                        p_val = result_dict['p_val']
                        r2 = result_dict['R2']

                        label = f'Pearson R: {pearson_corr: .2g} p = {pearson_p: .2g}'\
                        + '\n' + f'Slope: {slope: .2g} +/- {stderr: 0.2g} p = {p_val: .2g}'\
                        + '\n' f'R2: {r2: .2g}'

                        # Get algorithm name and RFE
                        alg_name = algorithm.estimator.__clas__.__name__
                        if algorithm.recursive_features:
                            rfe = f'RFE_{algorithm.recursive_features}'
                        else:
                            rfe = 'NoRFE'

                        # Plot
                        f1 = plt.figure()
                        plt.plot(ad_measure_array, pred_error, 'ro', label=label)
                        plt.xlabel(key)
                        plt.ylabel(error_metric)
                        plt.title(alg_name)
                        plt.legend(loc="upper left")

                        # Save
                        file_name = f'{key}_test_set{i}.png'
                        file_path = os.path.join(out_path, alg_name, rfe, error_metric, 'Correlations', file_name)
                        plt.tight_layout()
                        plt.savefig(file_path)
        
        return None

    def plot_correlations(self, algorithm, measure_type, ad_measure, error_metric, out_path):
        """ Plots correlations between prediction errors and AD measures.

        algorithm: Set to 'RF', 'kNN', 'GP', 'GBT', or 'SVR' for desired algorithm
        measure_type: Set to 'distance' or 'boundary according to desired ad measure type
        ad_measure: Set to any of dataset.ad_measures.measure_type.keys() according to desired ad measure
        error_metric: Type of prediction error used in correlation calculations, can be 'Squared_error' or
                      'Absolute_error'
        out_path: Path for parent PNG folder. Algorithm name is append to the path automatically.
        
        """

        # Loop through all results
        for i in range(len(self.X_test)):

            # Get ad measure array
            if ad_measure == 'Intrinsic':
                ad_measure_array = self.score_dicts[algorithm][ad_measure]['arrays'][i]
            else:
                # Skip measure if empty array
                if not self.test_set_ad_measures[measure_type][ad_measure]:
                    print(f'No data for {measure_type}: {ad_measure}.')

                # Get array if it exists
                else:
                    ad_measure_array = self.test_set_ad_measures[measure_type][ad_measure][i]
            
                    # Get prediction error array
                    pred_error = self.score_dicts[algorithm]['Algorithm'][error_metric][i]

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
                    file_path = os.path.join(out_path, algorithm, error_metric, 'Correlations', file_name)
                    plt.tight_layout()
                    plt.savefig(file_path)

            return None

    def save_pkl(self, algorithm=None, out_path=None):
        """ Saves dataset and algorithm information in a PKL file.
        
        algorithm: Algorithm class object with predictions and correlation results
        out_path: Optional custom PKL file path
        """
        # Class dictionaries
        dataset_dict = self.__dict__.copy()
        algorithm_dict = algorithm.__dict__.copy()
        dict_list = [dataset_dict,algorithm_dict]

        # path variables
        alg_name = algorithm.estimator.__class__.__name__

        if algorithm.recursive_features:
            rfe = f'RFE_{algorithm.recursive_features}'
        else:
            rfe = 'NoRFE'

        if self.scaled_:
            scaled = self.scaled_
        else:
            scaled = 'None'

        if self.var_filter_['Value']:
            varfilt = self.var_filter_['Value']
        else:
            varfilt = 'None'

        if self.corr_filter_['Value']:
            corrfilt = self.corr_filter_['Value']
        else:
            corrfilt = 'None'

        dataset_name = self.dataset_name
        file_name = f'{dataset_name}_{alg_name}_{rfe}_Scaled{scaled}_Var{varfilt}_Corr{corrfilt}.pkl'

        if not out_path:
            pkl_folder = r'C:\Users\skolmar\PycharmProjects\Modeling\AD_Methods\PKL'
            out_path = os.path.join(pkl_folder, dataset_name, alg_name, file_name)
        else:
            out_path = os.path.join(out_path, dataset_name, alg_name)

        output = open(out_path, 'wb')
        pickle.dump(dict_list, output)
        output.close()

        return None

###################
# PKL METHOD
###################

def plot_comparisons_pkl(pkl_file, out_path):
        """ Plots differences and ratios of metrics for compounds inside and outside the applicability domain,
        for each AD measure, for a given algorithm.

        algorithm: Algorithm class object with predictions
        out_path: Absolute path to the results for a dataset
        """

        pkl_list = pickle.load(open(pkl_file, 'rb'))
        data_dict = pkl_list[0]
        alg_dict = pkl_list[1]

        # Loop through all test sets
        for i in range(len(data_dict['X_test'])):

            # Loop through all measure types
            for ad_measure_type in ['distance', 'boundary']:
                
                # Loop through each measure
                for key in data_dict['test_set_ad_measures'][ad_measure_type].keys():
                    
                    # Get ratios
                    ratio_dict =  alg_dict['Ratios'][ad_measure_type][key][i]
                    rmse_ratio = ratio_dict['rmse_ratios']
                    r2_ratio = ratio_dict['r2_ratios']
                    rmse_ratio_label = 'RMSEin/RMSEout'
                    r2_ratio_label = 'R2in/R2out'

                    # Get differences
                    difference_dict = alg_dict['Differences'][ad_measure_type][key][i]
                    rmse_difference = difference_dict['rmse_differences']
                    r2_difference = difference_dict['r2_differences']
                    rmse_difference_label = 'RMSEout-RMSEin'
                    r2_difference_label = 'R2in-R2out'
            

                    # Set xaxis
                    if ad_measure_type == 'distance':
                        ratio_xaxis = ratio_dict['threshold_percentiles']
                        ratio_xlabel = 'Training Set Distance Percentile'
                        difference_xaxis = difference_dict['threshold_percentiles']
                        difference_xlabel = 'Training Set Distance Percentile'
                    elif ad_measure_type == 'boundary':
                        ratio_xaxis = ratio_dict['thresholds']
                        ratio_xlabel = 'Thresholds'
                        difference_xaxis = difference_dict['thresholds']
                        difference_xlabel = 'Thresholds'

                    # Make Ratio figure with two different y-axes on same plot
                    fig, ax1 = plt.subplots()

                    # Plot Ratio R2 data
                    color = 'tab:red'
                    ax1.set_xlabel(ratio_xlabel)
                    ax1.set_ylabel(r2_ratio_label, color=color)
                    ax1.plot(ratio_xaxis, r2_ratio, color=color)
                    ax1.tick_params(axis='y', labelcolor=color)

                    # Set duplicate x axis
                    ax2 = ax1.twinx()

                    # Plot Ratio RMSE data
                    color = 'tab:blue'
                    ax2.set_ylabel(rmse_ratio_label, color=color)
                    ax2.plot(ratio_xaxis, rmse_ratio, color=color)
                    ax2.tick_params(axis='y', labelcolor=color)

                    # Set figure title and layout
                    alg_name = alg_dict['estimator'].__class__.__name__
                    fig.suptitle(f'{alg_name}, {key}')
                    fig.tight_layout()

                    # Save figure
                    if alg_dict['recursive_features']:
                        rfe = f"RFE_{alg_dict['recursive_features']}"
                    else:
                        rfe = 'NoRFE'
                    file_name = f'Ratios_{key}_test_set{i}.png'
                    file_path = os.path.join(out_path, alg_name, rfe,'Ratios', file_name)
                    plt.savefig(file_path)

                    # Make Difference figure
                    fig2, ax3 = plt.subplots()

                    # Plot Difference R2 data
                    color = 'tab:red'
                    ax3.set_xlabel(difference_xlabel)
                    ax3.set_ylabel(r2_difference_label, color=color)
                    ax3.plot(difference_xaxis, r2_difference, color=color)
                    ax3.tick_params(axis='y', labelcolor=color)

                    # Set duplicate x axis
                    ax4 = ax3.twinx()

                    # Plot Ratio RMSE data
                    color = 'tab:blue'
                    ax4.set_ylabel(rmse_difference_label, color=color)
                    ax4.plot(difference_xaxis, rmse_difference, color=color)
                    ax4.tick_params(axis='y', labelcolor=color)

                    # Set figure title and layout
                    fig.suptitle(f'{alg_name}, {key}')
                    fig.tight_layout()

                    # Save figure
                    file_name = f'Differences_{key}_test_set{i}.png'
                    file_path = os.path.join(out_path, alg_name, rfe, 'Differences', file_name)
                    plt.savefig(file_path)

        return None

def pkl_plot_correlations(pkl_file, out_path):
        """
        algorithm: Algorithm class object with predictions and correlations
        out_path: Absolute path to PNG folder
        """

        pkl_list = pickle.load(open(pkl_file, 'rb'))
        data_dict = pkl_list[0]
        alg_dict = pkl_list[1]
        
        # Loop through each test set
        for i in range(len(data_dict['X_test'])):
            
            # Loop through each error metric
            for error_metric in ['squared_error', 'absolute_error']:

                # Get prediction error
                pred_error = alg_dict[error_metric][i]

                # Loop through each measure type
                for ad_measure_type in ['distance', 'boundary']:
                    
                    # Loop through each measure
                    for key in alg_dict['Results'][error_metric][ad_measure_type].keys():
                        
                        # Get ad measure array
                        ad_measure_array = data_dict['test_set_ad_measures'][ad_measure_type][key][i]
                        
                        # Get result dictionary
                        result_dict = alg_dict['Results'][error_metric][ad_measure_type][key][i]
                        pearson_corr = result_dict['Pearson_correlation']
                        pearson_p = result_dict['Pearson_p_val']
                        slope = result_dict['Slope']
                        stderr = result_dict['Stderr']
                        p_val = result_dict['p_val']
                        r2 = result_dict['R2']

                        label = f'Pearson R: {pearson_corr: .2g} p = {pearson_p: .2g}'\
                        + '\n' + f'Slope: {slope: .2g} +/- {stderr: 0.2g} p = {p_val: .2g}'\
                        + '\n' f'R2: {r2: .2g}'

                        # Get algorithm name and RFE
                        alg_name = alg_dict['estimator'].__class__.__name__
                        if alg_dict['recursive_features']:
                            rfe = f"RFE_{alg_dict['recursive_features']}"
                        else:
                            rfe = 'NoRFE'

                        # Plot
                        f1 = plt.figure()
                        plt.plot(ad_measure_array, pred_error, 'ro', label=label)
                        plt.xlabel(key)
                        plt.ylabel(error_metric)
                        plt.title(alg_name)
                        plt.legend(loc="upper left")

                        # Save
                        file_name = f'{key}_test_set{i}.png'
                        file_path = os.path.join(out_path, alg_name, rfe, error_metric, 'Correlations', file_name)
                        plt.tight_layout()
                        plt.savefig(file_path)
        
        return None