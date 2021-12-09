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