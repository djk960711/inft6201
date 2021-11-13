import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from scipy.stats.mstats import winsorize
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import kruskal, mannwhitneyu
from sklearn.cluster import AgglomerativeClustering
from typing import Dict


class Feature:
    """
    This module is designed to house all functionality related to feature creation and clustering analysis
    """
    @staticmethod
    def create_rct_column(ingested_file: pd.DataFrame, feature_config: Dict) -> pd.Series:
        """
        :param ingested_file: The dataset after initial preprocessing
        :return: A column with Roadway Clearance Time, and winsorizing applied to trim outliers
        """
        roadway_clearance_time = (ingested_file['End_Time'] - ingested_file['Start_Time']).dt.total_seconds() / 60
        roadway_clearance_time = winsorize(
            roadway_clearance_time,
            limits=feature_config['roadway_clearance_time']['winsorizing_ratio']
        )
        return roadway_clearance_time

    @staticmethod
    def create_road_type_column(ingested_file: pd.DataFrame, feature_config: Dict):
        """
        This uses regex to extract the road type from the Street column.
        :param ingested_file:
        :param feature_config: The config containing the list of types and the regex to map to each (e.g. ST -> Street)
        :return: Road type column
        """
        return np.select(
            [
                ingested_file["Street"].str.contains(
                    # Look for words indicating a street, boulevard or lane
                    re.compile(
                        regex_rule,
                        re.IGNORECASE
                    ),
                    regex=True
                )
                for regex_rule
                in feature_config['street_type']['rules'].values()
            ],
            feature_config['street_type']['rules'].keys(),
            default=feature_config['street_type']['default']
        )


    @staticmethod
    def created_incident_type_column(ingested_file: pd.DataFrame, feature_config: Dict):
        """
        This performs regex on the Description column to extract whether the incident is a specific type (e.g. Truck)
        :param ingested_file:
        :param feature_config: The incident type and the regex to map to each. (E.g. \bTRUCK\b -> Truck)
        :return: The column with a binary match.
        """
        return pd.concat([
            pd.Series(
                ingested_file["Description"].str.contains(
                    # Look for words indicating an incident type
                    re.compile(
                        regex_rule,
                        re.IGNORECASE
                    ),
                    regex=True
                ),
                name=column,
                dtype=np.int32
            )
            for column, regex_rule
            in feature_config['incident_type'].items()
        ], axis=1)

    @staticmethod
    def created_simp_weather_column(ingested_file: pd.DataFrame, feature_config: Dict):
        """
        This simplifies the numerous weather values into a series of simple flags
        :param ingested_file:
        :param feature_config: The weather categories and the weather values to use in each
        :return: The column with a binary match.
        """
        return pd.concat([
            pd.Series(
                ingested_file["Weather_Condition"].isin(conditions_to_include),
                name=column,
                dtype=np.int32
            )
            for column, conditions_to_include
            in feature_config['Weather_Simplified'].items()
        ], axis=1)

    @staticmethod
    def created_hour_type_column(ingested_file: pd.DataFrame, feature_config: Dict):
        """
        This matches to the hour of crash starting to classify the time-of-day
        (where categories based on a cluster analysis)
        :param ingested_file:
        :param feature_config: The time-of-day category and the hours in this category
        :return: The column with a binary match.
        """
        return pd.concat([
            pd.Series(
                ingested_file["Hour"].isin(included_hours),
                name=column,
                dtype=np.int32
            )
            for column, included_hours
            in feature_config['time_of_day'].items()
        ], axis=1)

    @staticmethod
    def created_clustered_column(
            ingested_file: pd.DataFrame,
            source_column: str,
            feature_config: Dict
    ) -> [(plt.Figure, pd.DataFrame, pd.Series)]:
        """
        This method calculates the proportion of each severity group by category, then uses this to group similar
        clusters together through hierarchical clustering
        :param ingested_file:
        :param feature_config: Details on the clustering specific to perform
        :return: The dendrogram figure, the mapping from type to cluster and the new column
        """
        # Return the proportion of severity for each of the weather conditions - this will become the clustering space
        grouped_severity = pd.concat([
            calculate_proportion_within_partition(data, "ID", [source_column, "Severity"])
            for _, data
            in ingested_file.groupby(by=[source_column])
        ]).pivot(
            index=source_column,
            columns="Severity",
            values="ID"
        ).fillna(0)

        # Use this space to cluster and create a figure
        dend_model = AgglomerativeClustering(
            distance_threshold=0,
            n_clusters=None,
            linkage=feature_config[source_column]['algorithm']
        )
        fig, axes = plt.subplots(1, 1, figsize=feature_config[source_column]['fig_size'])
        dend_model = dend_model.fit(grouped_severity)
        plt.title(f"Hierarchical Clustering of {source_column} by Severity")
        plot_dendrogram( # Plot the dendrogram, and label the categories
            dend_model,
            labels=[str(category).replace("_", " ") for category in grouped_severity.index],
            leaf_font_size=feature_config[source_column]['font_size']
        )
        plt.xticks(rotation=45)
        plt.xlabel(source_column)

        # Cut the tree and report the clustered weather conditions
        cut_tree = AgglomerativeClustering(
            n_clusters=feature_config[source_column]['number_clusters'],
            linkage=feature_config[source_column]['algorithm']
        )
        cut_tree.fit_predict(grouped_severity)
        # Create a dictionary from original category to cluster
        cluster_mapping = dict(zip(grouped_severity.index, cut_tree.labels_))
        # Apply the readable names to these clusters
        cluster_mapping = {
            original_weather_type: feature_config[source_column]['cluster_names'][cluster_index]
            for original_weather_type, cluster_index
            in cluster_mapping.items()
        }
        # Create a new column with the clustered categories
        new_column = ingested_file[source_column].replace(
            cluster_mapping
        ).fillna(feature_config[source_column]['default_cluster'])
        return (fig, pd.DataFrame.from_dict(cluster_mapping, orient='index', columns=['cluster']), new_column)
    @staticmethod
    def create_regional_council_column(
            ingested_file: pd.DataFrame,
            feature_config: Dict
    ) -> pd.Series:
        mappings = { # Get a mapping from county to regional council from the config.
            county: regional_council
            for regional_council, counties
            in feature_config['Regional_Council'].items()
            for county in counties
        }
        return ingested_file['County'].replace(mappings)

    @staticmethod
    def perform_kw_test(data: pd.DataFrame, columns: list[str]):
        """
        Performs the kruskal wallis test on all columns and returns a dataframe for all tests
        :param data: The original dataset
        :param columns: The set of columns in which to perform the test
        :return: A dataframe with the test results for all columns
        """
        # Filter out data beyond June 2020
        data = data[~((data["Start_Time"].dt.month > 7) & (data["Start_Time"].dt.year == 2020))]
        # Perform tests iteratively
        test_outcomes = [
            [column, *kruskal(*data.groupby(by=[column])['Severity'].apply(list))]
            for column in columns # Run for each column
            if len(data[column].unique())>1 # Ensure that the column has more than 1 value, otherwise the test is undef.
        ]
        # Convert to df and return.
        return pd.DataFrame.from_records(test_outcomes, columns=['Feature', 'Test Statistic', 'pvalue'])

    @staticmethod
    def perform_mw_test(data: pd.DataFrame, columns: list[str]):
        """
        Performs the mann-whitney u test on all pairs of categories in specified columns and returns a dataframe for
            all tests
        :param data: The original dataset
        :param columns: The set of columns in which to perform the test
        :return: A dataframe with the test results for all pairs in all columns
        """
        # Filter out data beyond June 2020
        data = data[~((data["Start_Time"].dt.month > 7) & (data["Start_Time"].dt.year == 2020))]
        # Perform tests iteratively
        test_outcomes = [
            [column, pair1, pair2,
                *mannwhitneyu(data.loc[data[column] == pair1, 'Severity'], data.loc[data[column] == pair2, 'Severity'])]
            for column in columns  # Run for each column
            if len(data[column].unique()) > 1 #Ensure that the column has more than 1 value, otherwise the test is undef.
            for pair1 in data[column].unique()
            for pair2 in data[column].unique()
            if pair1 > pair2 # Only take one set of pairs once
        ]
        # Convert to df and return.
        return pd.DataFrame.from_records(
            test_outcomes,
            columns=['Feature', 'Category 1', 'Category 2', 'Test Statistic', 'pvalue']
        )


def plot_dendrogram(model, **kwargs):
    """
    This is from the sklearn docs and creates a scikit dendrogram from a sklearn model
    https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    :param kwargs:
    :return:
    """
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)


def calculate_proportion_within_partition(df, index, groupby):
    """
    This util function calculates the proportion of
    :param df:
    :param index:
    :param groupby:
    :return:
    """
    total_records = df.count()[index]
    return (df.groupby(by=groupby)[index].count() / total_records).reset_index()
