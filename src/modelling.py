import math
from typing import Dict
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model, svm, metrics

class Modelling:
    """
    This module is designed to house all functionality related to feature imputing, model fitting and diagnostics
    """
    @staticmethod
    def impute_missing_values(feature_ready_file: pd.DataFrame, modelling_config: Dict) -> pd.DataFrame:
        """

        :param feature_ready_file: The dataset
        :param modelling_config: The config, specifying which columns to impute and which method to use.
        :return: A dataset with all columns imputed.
        """
        global impute_function
        columns_to_impute = modelling_config['columns_to_impute']
        for column, impute_method in columns_to_impute.items():
            # Firstly determines the impute function to use based on the config
            if impute_method=="median":
                impute_function = np.nanmedian
            elif impute_method=="mean":
                impute_function = np.nanmean
            elif impute_method=="mode":
                impute_function = lambda x: stats.mode(x)[0][0]
            # Applies the imputation below.
            feature_ready_file[column] = feature_ready_file[column].fillna(impute_function(feature_ready_file[column]))
        return feature_ready_file

    @staticmethod
    def fit_cv_model(
        X,
        Y,
        alpha_min = 1e0,
        alpha_max=3e3,
        n_iterations=25,
        n_folds=5,
        chosen_alpha = None
    ) -> [(pd.DataFrame, pd.DataFrame, float, LogisticRegression, StandardScaler)]:
        """
        This performs the CV fitting and outputs the chosen model.
        :param n_folds: The number of folds for CV
        :param alpha_max: The maximum regularisation parameter to search
        :param chosen_alpha: The selected regularisation parameter to output, if known
        :param X: The training values
        :param Y: The training labels
        :param alpha_min: The minimum alpha to search
        :param n_iterations: The number of alpha values to search
        :return: The results from each iteration, the CV plot and the preferred model.
        """
        # Firstly standardises the predictors, a requirement for regularisation.
        X_norm, normaliser = normalise_and_rebalance(X)
        Y = np.array(Y)
        # Initiate k-folds, stratified to maintain class balance (Avoid having to use SMOTE on one set due to imbalance)
        k_folds = StratifiedKFold(n_splits=n_folds)
        alphas = [ # Initiate a series of alpha (lambda) values based on a logarithmic range.
            10 ** x
            for x
            in np.arange(math.log10(alpha_min), math.log10(alpha_max), (math.log10(alpha_max)-math.log10(alpha_min)) / n_iterations)
        ]
        # Creates a dataframe with the AUC ROC score for each fold and each alpha (lambda) regularisation parameter.
        roc_scores = []
        number_of_selected_features = []
        for train_index, test_index in k_folds.split(X_norm, Y):
            roc_scores_within_fold = []
            number_of_selected_features_within_fold = []
            for alpha in alphas:
                model = fit_model(X_norm[train_index], Y[train_index], alpha)
                roc_scores_within_fold.append(
                    metrics.roc_auc_score(
                        Y[test_index],
                        model.predict_proba(X_norm[test_index])[:, 1]
                    )
                )
                number_of_selected_features_within_fold.append(
                    len(model.coef_[model.coef_!=0])
                )
            roc_scores.append(roc_scores_within_fold)
            number_of_selected_features.append(number_of_selected_features_within_fold)
        roc_scores = pd.DataFrame(np.transpose(np.stack(roc_scores)), index=alphas)
        number_of_selected_features = pd.DataFrame(np.transpose(np.stack(number_of_selected_features)), index=alphas)
        # Identfies the regularisation parameter with the highest average AUC ROC score.
        optimal_alpha = roc_scores.sum(axis=1).sort_values(ascending=False).index[0]
        # Returns the chosen model if a model has been chosen, otherwise the optimal based on the CV.
        optimal_model = fit_model(X_norm, Y, alpha=chosen_alpha if chosen_alpha else optimal_alpha)
        return roc_scores, number_of_selected_features, optimal_alpha, optimal_model, normaliser

    @staticmethod
    def compute_cv_curve(
            model_error_values: pd.DataFrame,
            number_of_features: pd.DataFrame,
            optimal_alpha: float,
            chosen_alpha: float
    ):
        """
        Plots how the mean-squared error varies with lambda parameter.
        :param model:
        :return: The figure to be saved
        """
        alphas = model_error_values.index # Get the list of regularisation parameters
        mean_auc_for_alpha = model_error_values.mean(axis=1) # Get the mean AUC value for each regularisation parameter
        number_of_features_for_alpha = number_of_features.mode(axis=1).median(axis=1)
        # Get the error bounds on each AUC value
        minus_sd_auc_for_alpha = [mu-sd for mu, sd in zip(list(mean_auc_for_alpha), list(model_error_values.std(axis=1)))]
        pos_sd_auc_for_alpha = [mu+sd for mu, sd in zip(list(mean_auc_for_alpha), list(model_error_values.std(axis=1)))]
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12,6))
        plt.plot(alphas, mean_auc_for_alpha) # Plot the alpha vs mean AUC
        plt.fill_between( # Add in the error bounds
            alphas,
            minus_sd_auc_for_alpha,
            pos_sd_auc_for_alpha,
            alpha=0.2
        )
        plt.vlines( # Plot a vertical line at the optimal parameter
            optimal_alpha,
            min(minus_sd_auc_for_alpha),
            max(pos_sd_auc_for_alpha),
            linestyles="dashed",
            label="Highest AUC"
        )
        plt.vlines( # Plot a vertical line at the selected parameter
            chosen_alpha,
            min(minus_sd_auc_for_alpha),
            max(pos_sd_auc_for_alpha),
            linestyles="dashed",
            label="Selected value",
            colors='r' # Make this line red to differentiate
        )
        [
            ax.annotate(
                f'{number_of_features_for_alpha[alpha]:.0f} features',
                [alpha, mean_auc_for_alpha[alpha]],
                ha='right',
                va='top',
                rotation=45
            )
            for alpha
            in alphas
        ]
        # Formatting
        ax.set_xscale('log')
        plt.legend()
        ax.set_xlabel("Lambda Value")
        ax.set_ylabel("AUC-ROC score")
        ax.set_title("AUC-ROC score for different regularisation parameters")
        return fig

    @staticmethod
    def create_auc_plot(model: LogisticRegression, X, Y):
        """
        This takes the logistic regression model and computes the AUC plot based on the validation data
        :param model: The logistic regression model
        :param X: The predictors
        :param Y: The response
        :return: A matplotlib plot of the ROC curve
        """
        # Get the roc curve metrics
        fpr, tpr, _ = metrics.roc_curve(Y, model.predict_proba(X)[:,1])
        fig, ax = plt.subplots(1,1,figsize=(10,8))
        plt.plot([0,1], [0,1], linestyle='--', label='Random selection model')
        plt.plot(fpr, tpr, marker='.', label='Logistic')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.title("AUC-ROC curve for the Severity model fit")
        return fig


def normalise_and_rebalance(X):
    normaliser = StandardScaler()
    X_norm = normaliser.fit_transform(X)

    return X_norm, normaliser


def fit_model(train_X, train_Y, alpha=1) -> LogisticRegression:
    """
    This function performs the model fit for a given regularisation parameter and given set of training folds.
    :param train_X:
    :param train_Y:
    :param alpha:
    :return:
    """
    # Initialise model using saga as the solver, since this solver supports LASSO
    model = LogisticRegression(penalty='l1', C=1/alpha, solver='saga', max_iter=400, random_state=42)
    # Oversample the training data
    oversample = SMOTE(random_state=42)
    X, Y = oversample.fit_resample(train_X, train_Y)
    # Fit the model based on the oversampled data.
    model.fit(X,Y)
    return model