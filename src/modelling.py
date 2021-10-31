from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from .OrdLassoCV import OrdinalLasso

import matplotlib.pyplot as plt

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
        normaliser = StandardScaler()
        train_X_norm = normaliser.fit_transform(train_X)
        oversample = SMOTE()
        X, Y = oversample.fit_resample(train_X_norm, train_Y)
        model = OrdinalLasso(cv=10, max_iter=10000, eps=1e-2)
        model.fit(X,Y)
        return model, normaliser

    @staticmethod
    def compute_cv_curve(model: OrdinalLasso):
        """
        Plots how the mean-squared error varies with lambda parameter.
        :param model:
        :return: The figure to be saved
        """
        alphas = model.alphas_
        mean_mse_for_alpha = [np.mean(x) for x in model.mse_path_]
        minus_sd_mse_for_alpha = [np.mean(x)-np.std(x) for x in model.mse_path_]
        pos_sd_mse_for_alpha = [np.mean(x) + np.std(x) for x in model.mse_path_]
        fig, ax = plt.subplots(1, 1, figsize=(12,6))
        plt.plot(alphas, mean_mse_for_alpha)
        plt.fill_between(
            alphas,
            minus_sd_mse_for_alpha,
            pos_sd_mse_for_alpha,
            alpha=0.2
        )
        plt.vlines(
            model.alpha_,
            0,
            max(pos_sd_mse_for_alpha),
            linestyles="dashed"
        )
        ax.set_xscale('log')
        ax.set_xlabel("Lambda Value")
        ax.set_ylabel("Mean Squared Error (MSE)")
        ax.set_title("Mean Squared Error (MSE) for different regularisation parameters")
        return fig
# def get_imputation_function(impute_method, data):
#     if impute_method=="median":
#         return