import os
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from src import Ingestion, Feature, Modelling

def ingestion_and_enrichment():
    """
    Runs the pipeline from ingestion to preprocessing
    :return: Outputs the preprocessed data for use in modelling
    """
    with open("conf.yaml", 'r') as config_file:
        # Read config file
        config = yaml.safe_load(config_file)
    # Step 1: Preprocessing
    raw_data = pd.read_csv(config['input_file'])
    raw_data = Ingestion.parse_datetime_columns(raw_data, config['ingestion'])
    ingested_data = Ingestion.dedupe_asset(raw_data, config['ingestion'])

    # Step 2: Feature Creation
    ingested_data['Roadway_Clearance_Time'] = Feature.create_rct_column(ingested_data, config['feature_creation'])
    ingested_data['Road_Type'] = Feature.create_road_type_column(ingested_data, config['feature_creation'])
    ingested_data['Hour'] = ingested_data['Start_Time'].dt.hour
    hour_plot, hour_mappings, ingested_data['Hour_Cluster'] = Feature.created_clustered_column(
        ingested_data,
        'Hour',
        config['feature_creation']
    )
    weather_plot, weather_mappings, ingested_data['Weather_Cluster'] = Feature.created_clustered_column(
        ingested_data,
        'Weather_Condition',
        config['feature_creation']
    )
    county_plot, county_mappings, ingested_data['County_Cluster'] = Feature.created_clustered_column(
        ingested_data,
        'County',
        config['feature_creation']
    )
    road_type_plot, road_type_mappings, ingested_data['Road_Type_Cluster'] = Feature.created_clustered_column(
        ingested_data,
        'Road_Type',
        config['feature_creation']
    )
    # Step n: Save results to the results folder
    if not os.path.exists("results/preprocessing"):
        os.makedirs("results/preprocessing")
    ingested_data.to_csv("results/preprocessing/ingested_data.csv")
    weather_mappings.to_csv("results/preprocessing/weather_mappings.csv")
    weather_plot.savefig("results/preprocessing/weather_clustering.png", bbox_inches='tight')
    county_mappings.to_csv("results/preprocessing/county_mappings.csv")
    county_plot.savefig("results/preprocessing/county_clustering.png", bbox_inches='tight')
    road_type_mappings.to_csv("results/preprocessing/road_type_mappings.csv")
    road_type_plot.savefig("results/preprocessing/road_type_clustering.png", bbox_inches='tight')
    hour_mappings.to_csv("results/preprocessing/hour_mappings.csv")
    hour_plot.savefig("results/preprocessing/hour_clustering.png", bbox_inches='tight')

def modelling_and_diagnostics():
    """
    Runs the pipeline from modelling to diagnostics
    :return: Outputs the preprocessed data for use in modelling
    """
    with open("conf.yaml", 'r') as config_file:
        # Read config file
        config = yaml.safe_load(config_file)
    ingested_data = pd.read_csv("results/preprocessing/ingested_data.csv")
    ingested_data = Ingestion.parse_datetime_columns(ingested_data, config['ingestion'])

    # Step 3: Get dummies and perform any required imputation. Then subset columns and split into test / train
    enriched_data = Modelling.impute_missing_values(ingested_data, config['modelling'])
    enriched_data = pd.get_dummies(enriched_data, columns=config['modelling']['columns_to_dummy'], drop_first=True)

    enriched_data = enriched_data[[config['modelling']['response'], *config['modelling']['predictors']]]
    train_data, test_data = train_test_split(enriched_data, test_size=config['modelling']['test_split'])

    # Step 4: Fit Regularised Linear Regression model
    train_data_X = train_data[config['modelling']['predictors']]
    train_data_Y = train_data[config['modelling']['response']]
    validation_data_X = test_data[config['modelling']['predictors']]
    validation_data_Y = test_data[config['modelling']['response']]
    model_error_values, optimal_alpha, model, normaliser = Modelling.fit_cv_model(train_data_X, train_data_Y)
    print(f'Train error is {model.score(normaliser.transform(train_data_X), train_data_Y)} and'
          f' test error is {model.score(normaliser.transform(validation_data_X), validation_data_Y)}')
    coefficients = pd.DataFrame(zip(train_data_X.columns, model.coef_), columns=["predict", "coef_estimate"])
    conf_mat = confusion_matrix(validation_data_Y, model.predict(normaliser.transform(validation_data_X)))
    cv_fig = Modelling.compute_cv_curve(model_error_values, optimal_alpha)

    # Step n: Save results to the results folder
    if not os.path.exists("results/modelling"):
        os.makedirs("results/modelling")
    coefficients.to_csv("results/modelling/coef_estimates.csv")
    cv_fig.savefig("results/modelling/cv_fig.png", bbox_inches="tight")
    pd.DataFrame(conf_mat, columns=["1", "2", "3", "4"]).to_csv("results/modelling/conf_mat")


if __name__ == "__main__":
    # These two parts of the pipeline are split into two, since each is time-consuming.
    # ingestion_and_enrichment()
    modelling_and_diagnostics()
