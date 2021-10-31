from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from mord import LAD
from sklearn.preprocessing import Normalizer



class Modelling:
    @staticmethod
    def impute_missing_values(feature_ready_file: pd.DataFrame, modelling_config: Dict) -> pd.DataFrame:
        global impute_function
        columns_to_impute = modelling_config['columns_to_impute']
        for column, impute_method in columns_to_impute.items():
            if impute_method=="median":
                impute_function = np.nanmedian
            elif impute_method=="mean":
                impute_function = np.nanmean
            elif impute_method=="mode":
                impute_function = lambda x: stats.mode(x)[0][0]

            feature_ready_file[column] = feature_ready_file[column].fillna(impute_function(feature_ready_file[column]))
        return feature_ready_file

    @staticmethod
    def fit_model(train_X, train_Y):
        normaliser = Normalizer()
        train_X_norm = normaliser.fit_transform(train_X)
        model = LAD()
        model.fit(train_X_norm, train_Y)
        return model, normaliser
# def get_imputation_function(impute_method, data):
#     if impute_method=="median":
#         return