import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from scipy.stats.mstats import winsorize
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from typing import Dict


class Feature:
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
    def create_road_type_column(ingested_file: pd.DataFrame, feature_config: Dict) -> pd.Series:
        """
        This uses regex to extract the road type from the Street column.
        :param ingested_file:
        :param feature_config: The config containing the list of types and the regex to map to each (e.g. ST -> Street)
        :return: Road type column
        """
        return pd.Series(np.select(
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
        plot_dendrogram(
            dend_model,
            labels=grouped_severity.index,
            leaf_font_size=feature_config[source_column]['font_size']
        )
        plt.xlabel(source_column)

        # Cut the tree and report the clustered weather conditions
        cut_model = cut_tree = AgglomerativeClustering(
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
