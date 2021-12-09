import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE

#####################
# Algorithm CLASS
#####################

class Algorithm():

    def __init__(self, estimator, recursive_features):

        # Distance function list
        self.function_list = ['cosine_distance',
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

        # Boundary function list
        self.boundary_list = ['one_class_svm',
                        'robust_covariance',
                        'isolation_forest',
                        'local_outlier_factor']

        # Perform RFE if parameter enabled
        self.recursive_features = recursive_features
        if recursive_features:
            rfe = RFE(estimator, n_features_to_select=recursive_features)
            self.estimator = rfe
        else:
            self.estimator = estimator
        self.name = self.estimator.__class__.__name__

        # Prediction and prediction error
        self.predictions = []
        self.squared_error = []
        self.absolute_error = []

        self.intrinsic = []
        
        # Correlation results
        self.Results = {
            'squared_error':{
                'distance':{k:[] for k in self.function_list},
                'boundary': {k:[] for k in self.boundary_list},
                'intrinsic': []
            },
            'absolute_error':{
                'distance':{k:[] for k in self.function_list},
                'boundary': {k:[] for k in self.boundary_list},
                'intrinsic': []
            }   
        }

        # Ratios
        self.Ratios = {
            'distance':{k:[] for k in self.function_list},
            'boundary': {k:[] for k in self.boundary_list},
            'intrinsic': []
            }
        
        # Differences
        self.Differences = {
            'distance':{k:[] for k in self.function_list},
            'boundary': {k:[] for k in self.boundary_list},
            'intrinsic': []
            }
        
    def get_intrinsic(self, X_test):
        """ Gets the intrinsic applicability domain measure for a given algorithm.
        
        X_test: Test set from a dataset object.

        """
        alg_name = self.estimator.__class__.__name__

        # Random Forest
        if alg_name == 'RandomForestRegressor':
            
            all_tree_preds = {}
            # Loop through each estimator
            for each_tree in range(self.estimator.n_estimators):

                # Get predictions
                tree_preds = self.estimator.estimators_[each_tree].predict(X_test)
                all_tree_preds[f'tree_{each_tree}'] = tree_preds
                
            # Calculate standard deviation of tree predictions for each test set entry
            tree_df = pd.DataFrame.from_dict(data=all_tree_preds, orient='columns')
            tree_pred_std = tree_df.std(axis=1)
            self.intrinsic.append(tree_pred_std.values)
        
        # Gradient Boosted Trees
        if alg_name == 'GradientBoostingRegressor':
            
            all_tree_preds = {}

            # Predict with each estimator (tree)
            for i_tree,tree in enumerate(self.estimator.estimators_):
                tree_preds = self.estimator.estimators_[i_tree][0].predict(X_test)
                all_tree_preds[f'tree_{i_tree}'] = tree_preds
            
            # Calculate standard deviation of tree predictions for each test set entry
            tree_df = pd.DataFrame.from_dict(data=all_tree_preds, orient='columns')
            tree_pred_std = tree_df.std(axis=1)
            self.intrinsic.append(tree_pred_std.values)
        
        # Gaussian Process
        if alg_name == 'GaussianProcessRegressor':
            y_pred, y_stds = self.estimator.predict(X=X_test, return_std=True)
            self.intrinsic.append(y_stds)

        return None