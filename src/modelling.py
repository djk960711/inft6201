import math
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
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
    def fit_cv_model(
        X,
        Y,
        alpha_min = 1e-3,
        alpha_max=10**(-0.8),
        n_iterations=25,
        n_folds=5
    ) -> [(pd.DataFrame, float, OrdinalLasso, StandardScaler)]:
        """
        The native CV search cannot score based on an ordinal-based loss function.
        :param X: The training values
        :param Y: The training labels
        :param alpha_min: The minimum alpha to search
        :param n_iterations: The number of alpha values to search
        :return: The results from each iteration, the CV plot and the preferred model.
        """
        X_norm, normaliser = normalise_and_rebalance(X)
        Y = np.array(Y)
        k_folds = KFold(n_splits=n_folds)
        alphas = [
            10 ** x
            for x
            in np.arange(math.log10(alpha_min), math.log10(alpha_max), (math.log10(alpha_max)-math.log10(alpha_min)) / n_iterations)
        ]
        model_error_values = pd.DataFrame(np.transpose(np.stack([
            [
            np.array(fit_model(X_norm[train_index], Y[train_index], alpha).score(X_norm[test_index], Y[test_index]))
            for alpha
            in alphas
            ]
            for train_index, test_index in k_folds.split(X_norm)
        ])), index=alphas)
        optimal_alpha = model_error_values.sum(axis=1).sort_values(ascending=False).index[0]
        optimal_model = fit_model(X_norm, Y, alpha=optimal_alpha)
        return model_error_values, optimal_alpha, optimal_model, normaliser




    @staticmethod
    def compute_cv_curve(model_error_values: pd.DataFrame, optimal_alpha: float):
        """
        Plots how the mean-squared error varies with lambda parameter.
        :param model:
        :return: The figure to be saved
        """
        alphas = model_error_values.index
        mean_mse_for_alpha = model_error_values.mean(axis=1)
        minus_sd_mse_for_alpha = [mu-sd for mu, sd in zip(list(mean_mse_for_alpha), list(model_error_values.std(axis=1)))]
        pos_sd_mse_for_alpha = [mu+sd for mu, sd in zip(list(mean_mse_for_alpha), list(model_error_values.std(axis=1)))]
        fig, ax = plt.subplots(1, 1, figsize=(12,6))
        plt.plot(alphas, mean_mse_for_alpha)
        plt.fill_between(
            alphas,
            minus_sd_mse_for_alpha,
            pos_sd_mse_for_alpha,
            alpha=0.2
        )
        plt.vlines(
            optimal_alpha,
            0,
            min(minus_sd_mse_for_alpha),
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


def normalise_and_rebalance(X):
    normaliser = StandardScaler()
    X_norm = normaliser.fit_transform(X)

    return X_norm, normaliser


def fit_model(train_X, train_Y, alpha=0) -> OrdinalLasso:
    model = OrdinalLasso(alpha=alpha)  #(cv=10, max_iter=10000, eps=1e-2)
    oversample = SMOTE()
    X, Y = oversample.fit_resample(train_X, train_Y)
    model.fit(X,Y)
    return model