import numpy as np
import pandas as pd


class Descriptive:
    @staticmethod
    def generate_descriptives(data: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a summary of the descriptive statistics for each column.
        :param data:
        :return:
        """
        describe_statistics = pd.concat([  # describe_statistics
            data[column].describe()
            for column
            in data.columns
        ], axis=1).transpose().reset_index().rename(columns={"index": "Column Name"})
        data_dictionary = pd.concat([
            pd.Series([
                np.select(
                    [
                        column == "ID" or column == "Unnamed: 0",
                        str(data[column].dtype) == "datetime64[ns]",
                        (str(data[column].dtype) == "int64") | (str(data[column].dtype) == "int32"),
                        str(data[column].dtype) == "bool",
                        str(data[column].dtype) == "float64",
                        str(data[column].dtype) == "object"
                    ],
                    [
                        "ID",
                        "Datetime",
                        "Categorical", # This depends on the context
                        "Boolean",
                        "Continuous",
                        "Categorical"
                    ],
                    default="Categorical"
                )
                for column
                in data.columns
            ], name="Data Type"),
            describe_statistics
            ],
            axis=1
        ).merge(
            definitions,
            left_on="Column Name",
            right_on="Column Name",
            how="left"
        )

        data_dictionary['Coverage'] = data_dictionary['count']/data_dictionary['count'].max()


        categorical_columns_to_pull_all_unique_values = data_dictionary.loc[
            (data_dictionary['Data Type']=="Categorical") & (data_dictionary["unique"]<=10),
            "Column Name"
        ]
        data_dictionary = data_dictionary.merge(
            get_list_categorical_values(data, categorical_columns_to_pull_all_unique_values),
            on="Column Name",
            how="left"
        )

        data_dictionary = data_dictionary[[
            "Column Name",
            "Description",
            "Data Type",
            "Coverage",
            "unique",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "Values"
        ]]

        data_dictionary.columns = [column.capitalize() for column in data_dictionary.columns]
        return data_dictionary

def get_list_categorical_values(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    This gets the full list of all values for categorical columns
    :param data: The data to perform the analysis on
    :param columns: Categorical columns to search
    :return: A list of values for the relevant columns
    """
    return pd.DataFrame.from_dict({
            index: [
                column,
                ", ".join(sorted(data[column].unique().astype(np.str0)))
            ]
            for index, column
            in enumerate(columns)
        },
        orient="index",
        columns=["Column Name", "Values"]
    )

