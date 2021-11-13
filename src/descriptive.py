import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


class Descriptive:
    """
    This module is designed to house all functionality related to descriptive statistics.
    This includes:
     - Generating metrics on type, coverage
     - Plotting of relationships
    """
    @staticmethod
    def generate_descriptives(data: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a summary of the descriptive statistics for each column.
        :param data:
        :return:
        """
        # Step 1: Generate the describe statistics provided by pandas directly
        describe_statistics = pd.concat([  # describe_statistics
            data[column].describe()
            for column
            in data.columns
        ], axis=1).transpose().reset_index().rename(columns={"index": "Column Name"})
        # Step 2: Append the column data type and column definitions as separate columns
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
        # Step 3: Append coverage (1-null)
        data_dictionary['Coverage'] = data_dictionary['count']/data_dictionary['count'].max()
        # Step 4: Append the list of categories for all categorical columns with less than (or equal to) 10 distinct
        #           categories
        categorical_columns_to_pull_all_unique_values = data_dictionary.loc[
            (data_dictionary['Data Type']=="Categorical") & (data_dictionary["unique"]<=10),
            "Column Name"
        ]
        data_dictionary = data_dictionary.merge(
            get_list_categorical_values(data, categorical_columns_to_pull_all_unique_values),
            on="Column Name",
            how="left"
        )
        # Step 5: Select relevant columns for output
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
        # Step 6: Capitalise column names for readibility.
        data_dictionary.columns = [column.capitalize() for column in data_dictionary.columns]
        return data_dictionary

    @staticmethod
    def get_severity_by_month_plot(data: pd.DataFrame, raw_data: pd.DataFrame, width=5) -> plt.Figure:
        """
        This gets the number of incidents by severity and by month
        :param data: The dataset after deduping
        :param raw_data: The dataset before deduping
        :return: A plot of severity over month.
        """
        colours = ['#CCFF99', '#FFFF99', '#FFB266', '#FF6666']
        fig, ax = plt.subplots(1,1, figsize=(12,6))
        # Create a month column that is the start datetime truncated to month.
        data["Month"] = data['Start_Time'].dt.date + pd.offsets.MonthBegin(-1)
        raw_data["Month"] = raw_data['Start_Time'].dt.date + pd.offsets.MonthBegin(-1)
        # Get the counts by month and severity
        count_by_severity_month = data.groupby(by=["Month", "Severity"])["ID"].count().reset_index()
        count_by_severity_month_raw = raw_data.groupby(by=["Month", "Severity"])["ID"].count().reset_index()
        for index, severity in enumerate(unique_severity_values := sorted(count_by_severity_month["Severity"].unique())):
            # Plot the duplicated count bars as dotted
            plt.bar(
                mdates.date2num(count_by_severity_month_raw.loc[count_by_severity_month["Severity"] == severity, "Month"]) -
                len(unique_severity_values) / 2 * width + width * index,
                count_by_severity_month_raw.loc[count_by_severity_month_raw["Severity"] == severity, "ID"],
                label=f'Severity {severity} (with duplicates)',
                width=width,
                edgecolor=colours[index],
                linestyle='--',
                color='white',
                align="center"
            )
            # Plot the deduped bars as filled in
            plt.bar(
                mdates.date2num(count_by_severity_month.loc[count_by_severity_month["Severity"]==severity, "Month"])-
                    len(unique_severity_values)/2*width+width*index,
                count_by_severity_month.loc[count_by_severity_month["Severity"] == severity, "ID"],
                label=f'Severity {severity} (After removing duplicates)',
                width=width,
                color=colours[index],
                align="center"
            )
        ax.xaxis_date()
        plt.legend()
        plt.title("Number of incidents by severity and month")
        plt.xlabel("Month")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        plt.ylabel("Number of incidents")
        return fig

    @staticmethod
    def get_road_type_by_month_plot(data: pd.DataFrame) -> plt.Figure:
        """
        This gets the number of incidents by road type and by month
        :param data: The dataset after deduping
        :return: A plot of severity over month.
        """
        # This is used to remap certain road types to easier names
        readable_road_mappings = {
            "County_Route": "County Route",
            "State_Route": "State Route",
            "Federal_Route": "Federal Route",
            "Bridge_Tunnel": "Bridges and Tunnels",
            "Boulevard_Place": "Boulevards and Places",
            "Numbered_Street": "Numbered Street",
            "Street": "General Street",
            "Mall_Lane": "Malls and Laneways",
            "Park_Causeway": "Parks and Causeways"
        }
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        # Create a month column that is the start datetime truncated to month.
        data["Month"] = data['Start_Time'].dt.date + pd.offsets.MonthBegin(-1)
        # Get the counts by month and road type in a pivoted form
        count_by_road_type_month = data.groupby(by=["Month", "Road_Type"])["ID"].count().unstack().fillna(0)
        current_position = 0
        for road_type in sorted(count_by_road_type_month.columns):
            plt.bar(
                count_by_road_type_month.index,
                count_by_road_type_month[road_type],
                label=readable_road_mappings.get(road_type, road_type), # Remap confusing names to readable labels
                align="center",
                bottom=current_position,
                width=10,
                linewidth=0
            )
            current_position = current_position + count_by_road_type_month[road_type]
        ax.xaxis_date()
        plt.legend(ncol=2)
        plt.title("Number of incidents by road type and month")
        plt.xlabel("Month")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        plt.ylabel("Number of incidents")
        return fig

    @staticmethod
    def get_council_region_by_month_plot(data: pd.DataFrame) -> plt.Figure:
        """
        This gets the number of incidents by Council Region and by month
        :param data: The dataset after deduping
        :return: A plot of severity over month.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        # Create a month column that is the start datetime truncated to month.
        data["Month"] = data['Start_Time'].dt.date + pd.offsets.MonthBegin(-1)
        # Get the counts by month and road type in a pivoted form
        count_by_council_region_month = data.groupby(by=["Month", "Regional_Council"])["ID"].count().unstack().fillna(0)
        current_position = 0
        for council_region in sorted(count_by_council_region_month.columns):
            plt.bar(
                count_by_council_region_month.index,
                count_by_council_region_month[council_region],
                label=council_region.replace("_", " "),  # Remove underscores from the region names
                align="center",
                bottom=current_position,
                width=10,
                linewidth=0
            )
            current_position = current_position + count_by_council_region_month[council_region]
        ax.xaxis_date()
        plt.legend(ncol=2)
        plt.title("Number of incidents by regional council and month")
        plt.xlabel("Month")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        plt.ylabel("Number of incidents")
        return fig

    @staticmethod
    def get_severity_by_category_plot(data: pd.DataFrame, categorical_column: str, width=1) -> plt.Figure:
        """
        This gets the number of incidents by severity and category
        :param data: The dataset after deduping
        :param categorical_column: The column to create the plot for
        :param prop: Should it calculate the proportion, or a count?
        :return: A plot of severity by category
        """
        colours = ['#CCFF99', '#FFFF99', '#FFB266', '#FF6666']
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        # Remove the period of July 2020 to December 2020.
        data = data[~((data["Start_Time"].dt.month > 7) & (data["Start_Time"].dt.year == 2020))]
        # Get the counts by severity and category in a pivoted form
        count_by_category_severity = data.groupby(by=[categorical_column, "Severity"])["ID"].count().unstack().fillna(0)

        prop_by_category_severity = count_by_category_severity.div(count_by_category_severity.sum(axis=1), axis=0)
        # This current_position is used to construct stacked bar charts.
        current_position_prop, current_position_count = 0, 0
        for index, severity in enumerate(sorted(count_by_category_severity.columns)):
            ax[0].bar(
                [str(category) for category in count_by_category_severity.index],
                count_by_category_severity[severity],
                label=f'Severity {severity}',  # Remove underscores and replace with spaces
                align="center",
                bottom=current_position_count,
                width=width,
                linewidth=0,
                color=colours[index]
            )
            ax[1].bar(
                [str(category) for category in prop_by_category_severity.index],
                prop_by_category_severity[severity],
                label=f'Severity {severity}',  # Remove underscores and replace with spaces
                align="center",
                bottom=current_position_prop,
                width=width,
                linewidth=0,
                color=colours[index]
            )
            current_position_count = current_position_count + count_by_category_severity[severity]
            current_position_prop = current_position_prop + prop_by_category_severity[severity]
        #ax[0].legend(ncol=2)
        ax[0].set_xticklabels([str(category) for category in count_by_category_severity.index], rotation=80)
        ax[0].set_title(f"Number of incidents by severity and {categorical_column}")
        ax[0].set_xlabel(categorical_column)
        ax[0].set_ylabel(f"Number of incidents")
        ax[1].legend(ncol=2)
        ax[1].set_xticklabels([str(category) for category in prop_by_category_severity.index], rotation=80)
        ax[1].set_title(f"Proportion of incidents by severity and {categorical_column}")
        ax[1].set_xlabel(categorical_column)
        ax[1].set_ylabel(f"Proportion of incidents")
        return fig

    @staticmethod
    def get_time_of_day_by_day_of_week(data: pd.DataFrame, width=0.2):
        """
        Plot the time of day (tod) by day of week (dow), with stacked bars by severity.
        :param width: Width of the bars
        :param data: The enriched dataset
        :return: A plot with the time of day by day of week chart.
        """
        # Colours
        colours = ['#0066CC', '#CCC000', '#CC6600', '#7F00FF']
        # Filter out data beyond June 2020
        data = data[~((data["Start_Time"].dt.month > 7) & (data["Start_Time"].dt.year == 2020))]
        # Get a pivoted table of day-of-week and time-of-day
        tod_by_dow_data = data.groupby(by=[
            data["Start_Time"].dt.strftime('%w. %A'),
            data[["Late_Night", "Morning_Peak", "Midday", "Afternoon_Peak"]].idxmax(axis=1)
        ])["ID"].count().unstack()
        # Get the same as above, but only for high severity crashes
        tod_by_dow_data_high_severity = data[data['Severity_Gt_2'] == 1].groupby(by=[
            data["Start_Time"].dt.strftime('%w. %A'),
            data[["Late_Night", "Morning_Peak", "Midday", "Afternoon_Peak"]].idxmax(axis=1)
        ])["ID"].count().unstack()
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        # Create a side-by-side column plot.
        for index, tod in enumerate(['Morning_Peak','Midday','Afternoon_Peak','Late_Night']):
            # Create a bar for each time-of-day
            plt.bar(
                [ # Create a bar plot with side-by-side
                    x-len(tod_by_dow_data.columns) / 2 * width + width * index
                    for x
                    in range(0, len(tod_by_dow_data.index))
                ],
                tod_by_dow_data[tod]-tod_by_dow_data_high_severity[tod],
                label=tod.replace("_", " ") + " - Low severity (<=2)", # Add time of day to the legend and remove underscores from text.
                width=width,
                align="center",
                color=colours[index],
                alpha=0.4,
                bottom=tod_by_dow_data_high_severity[tod]
            )
            plt.bar(
                [  # Create a bar plot with side-by-side
                    x - len(tod_by_dow_data_high_severity.columns) / 2 * width + width * index
                    for x
                    in range(0, len(tod_by_dow_data_high_severity.index))
                ],
                tod_by_dow_data_high_severity[tod],
                label=tod.replace("_", " ") + " - High severity (>2)",
                # Add time of day to the legend and remove underscores from text.
                width=width,
                align="center",
                color=colours[index],
            )
        plt.legend(ncol=2)
        plt.title(f"Number of incidents by time-of-day and day-of-week")
        plt.xlabel("Day-of-week")
        plt.ylabel("Number of incidents")
        plt.xticks(range(0, len(tod_by_dow_data.index)), tod_by_dow_data.index)
        return fig

    @staticmethod
    def get_road_type_by_day_of_week(data: pd.DataFrame, width=0.15):
        """
        Plot the road type by day of week (dow)
        :param width: Width of the bars
        :param data: The enriched dataset
        :return: A plot with the road type by day of week chart.
        """
        # Filter out data beyond June 2020
        data = data[~((data["Start_Time"].dt.month > 7) & (data["Start_Time"].dt.year == 2020))]
        # Get a pivoted table of day-of-week and road type
        rt_by_dow_data = data.groupby(by=[
            data["Start_Time"].dt.strftime('%w. %A'),
            data["Road_Type_Cluster"]
        ])["ID"].count().unstack()
        # Calculate proportion for given dow
        rt_by_dow_data = rt_by_dow_data.div(rt_by_dow_data.sum(axis=1), axis=0)
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        for index, rt in enumerate(rt_by_dow_data.columns):
            # Create a bar for each time-of-day
            plt.bar(
                [  # Create a bar plot with side-by-side
                    x - len(rt_by_dow_data.columns) / 2 * width + width * index + width/2
                    for x
                    in range(0, len(rt_by_dow_data.index))
                ],
                rt_by_dow_data[rt],
                label=rt.replace("_", " "),  # Add time of day to the legend and remove underscores from text.
                width=width,
                align="center"
            )
        plt.legend()
        plt.title(f"Proportion of incidents by road type and day-of-week")
        plt.xlabel("Day-of-week")
        plt.ylabel("Proportion of incidents (of incidents for day-of-week)")
        plt.xticks(range(0, len(rt_by_dow_data.index)), rt_by_dow_data.index)
        return fig

    @staticmethod
    def get_weather_count_by_severity(data: pd.DataFrame, width=0.8):
        """
        Plot the severity by weather condition (simplified). Note that these aren't mutually exclusive, so care had to
         be taken here
        :param width: Width of the bars
        :param data: The enriched dataset
        :return: A plot with the count of incidents by severity and by weather condition.
        """
        # Colour scheme to apply to the plot
        colours = ['#CCFF99', '#FFFF99', '#FFB266', '#FF6666']
        # Filter out data beyond June 2020
        data = data[~((data["Start_Time"].dt.month > 7) & (data["Start_Time"].dt.year == 2020))]
        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(12,  4))
        # Getting the count by weather condition (simplified) is a challenge because weather condition is not mutually
        #  exclusive - an incident can occur when it is both windy and rainy.
        count_by_category_severity = pd.concat([
            data[data[weather_condition] == 1].replace({weather_condition: {1: weather_condition}}).groupby(
                [weather_condition, 'Severity']
            )['ID'].count().unstack()
            for weather_condition
            in sorted(["Snow_Ice", "Fog_Haze", "Clouds", "Rain", "Windy", "Storm", "Clear"])
        ])
        # Calculate the proportion of incidents by weather condition.
        prop_by_category_severity = count_by_category_severity.div(count_by_category_severity.sum(axis=1), axis=0)
        current_position_prop, current_position_count = 0, 0
        for index, severity in enumerate(sorted(count_by_category_severity.columns)):
            ax[0].bar(
                [str(category).replace("_", " ") for category in count_by_category_severity.index],
                count_by_category_severity[severity],
                label=f'Severity {severity}',  # Remove underscores and replace with spaces
                align="center",
                bottom=current_position_count,
                width=width,
                linewidth=0,
                color=colours[index]
            )
            ax[1].bar(
                [str(category).replace("_", " ") for category in prop_by_category_severity.index],
                prop_by_category_severity[severity],
                label=f'Severity {severity}',  # Remove underscores and replace with spaces
                align="center",
                bottom=current_position_prop,
                width=width,
                linewidth=0,
                color=colours[index]
            )
            current_position_count = current_position_count + count_by_category_severity[severity]
            current_position_prop = current_position_prop + prop_by_category_severity[severity]
        ax[0].legend(ncol=2)
        ax[0].set_xticklabels([str(category) for category in count_by_category_severity.index], rotation=80)
        ax[0].set_title(f"Number of incidents by Severity and \nWeather Condition (Simplified)")
        ax[0].set_xlabel("Weather Condition (Simplified)")
        ax[0].set_ylabel(f"Number of incidents")
        # ax[1].legend(ncol=2)
        ax[1].set_xticklabels([str(category) for category in prop_by_category_severity.index], rotation=80)
        ax[1].set_title(f"Proportion of incidents by Severity and \nWeather Condition (Simplified)")
        ax[1].set_xlabel("Weather Condition (Simplified)")
        ax[1].set_ylabel(f"Proportion of incidents")
        return fig

    @staticmethod
    def get_weather_count_by_time_of_day(data: pd.DataFrame, width=0.8):
        """
        Plot the severity by weather condition (simplified). Note that these aren't mutually exclusive, so care had to
         be taken here
        :param width: Width of the bars
        :param data: The enriched dataset
        :return: A plot with the count of incidents by severity and by weather condition.
        """
        # Colour scheme to apply to the plot
        colours = ['#0066CC', '#CCC000', '#CC6600', '#7F00FF']
        # Filter out data beyond June 2020
        data = data[~((data["Start_Time"].dt.month > 7) & (data["Start_Time"].dt.year == 2020))]
        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # Getting the count by weather condition (simplified) is a challenge because weather condition is not mutually
        #  exclusive - an incident can occur when it is both windy and rainy.
        count_by_category_tod = pd.concat([
            (
                filtered_data:=data[data[weather_condition] == 1].replace({weather_condition: {1: weather_condition}})
            ).groupby(
                [
                    filtered_data[weather_condition],
                    filtered_data[["Late_Night", "Morning_Peak", "Midday", "Afternoon_Peak"]].idxmax(axis=1)
                ]
            )['ID'].count().unstack()
            for weather_condition
            in sorted(["Snow_Ice", "Fog_Haze", "Clouds", "Rain", "Windy", "Storm", "Clear"])
        ])
        # Calculate the proportion of incidents by weather condition.
        prop_by_category_tod = count_by_category_tod.div(count_by_category_tod.sum(axis=1), axis=0)
        current_position_prop, current_position_count = 0, 0
        for index, tod in enumerate(['Morning_Peak','Midday','Afternoon_Peak','Late_Night']):
            ax[0].bar(
                [str(category).replace("_", " ") for category in count_by_category_tod.index],
                count_by_category_tod[tod],
                label=tod,  # Remove underscores and replace with spaces
                align="center",
                bottom=current_position_count,
                width=width,
                linewidth=0,
                color=colours[index]
            )
            ax[1].bar(
                [str(category).replace("_", " ") for category in prop_by_category_tod.index],
                prop_by_category_tod[tod],
                label=tod,
                align="center",
                bottom=current_position_prop,
                width=width,
                linewidth=0,
                color=colours[index]
            )
            current_position_count = current_position_count + count_by_category_tod[tod]
            current_position_prop = current_position_prop + prop_by_category_tod[tod]
        ax[0].legend(ncol=2)
        ax[0].set_xticklabels([str(category) for category in count_by_category_tod.index], rotation=80)
        ax[0].set_title(f"Number of incidents by Time-of-day and \nWeather Condition (Simplified)")
        ax[0].set_xlabel("Weather Condition (Simplified)")
        ax[0].set_ylabel(f"Number of incidents")
        # ax[1].legend(ncol=2)
        ax[1].set_xticklabels([str(category) for category in prop_by_category_tod.index], rotation=80)
        ax[1].set_title(f"Proportion of incidents by Time-of-day and \nWeather Condition (Simplified)")
        ax[1].set_xlabel("Weather Condition (Simplified)")
        ax[1].set_ylabel(f"Proportion of incidents")
        return fig

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
                ", ".join(sorted(data[column].unique().astype(np.str0))) # Concat categories into a string joined by ,
            ]
            for index, column
            in enumerate(columns)
        },
        orient="index",
        columns=["Column Name", "Values"]
    )
