import os
import numpy as np
import pandas as pd
import yaml

# SKLearn is used for modelling and diagnostics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from src import Ingestion, Feature, Modelling, Desctiptive

# The code is structred into two steps:
# - Ingestion and Enrichment: This takes the raw data, dedupes, imputes nulls and creates new features.
# - Modelling and Diagnostics: This takes the enriched data to split into train/validation, perform modelling and output
#    diagnostics

def ingestion_and_enrichment():
    """
    Runs the pipeline from ingestion to preprocessing
    :return: Outputs the preprocessed data for use in modelling
    """
    # The parameters used in the model, such as feature design and deduplication keys are stored in the config file.
    with open("conf.yaml", 'r') as config_file:
        # Read config file
        config = yaml.safe_load(config_file)
    # Step 1: Preprocessing
    raw_data = pd.read_csv(config['input_file'])
    definitions = pd.read_csv(config['definition_file']) # This contains the definition of each feature in readable form.

    ## 1.1. Datetime columns parsed into the correct datetime format
    raw_data = Ingestion.parse_datetime_columns(raw_data, config['ingestion'])
    ## 1.2. The asset is deduped on different combinations of keys in the asset
    ingested_data = Ingestion.dedupe_asset(raw_data, config['ingestion'])
    ## 1.3. Write out a descriptive statistics table for the report, including coverage and categorical values
    if not os.path.exists("results/descriptives"):
        os.makedirs("results/descriptives")
    Desctiptive.generate_descriptives(ingested_data, definitions).to_csv("results/descriptives/raw_deduped.csv")

    # Step 2: Feature Creation
    ## 2.1. Roadway clearance time: The time taken for the incident to be cleared
    ingested_data['Roadway_Clearance_Time'] = Feature.create_rct_column(ingested_data, config['feature_creation'])
    ## 2.2. Road and incident type: Using regex to extract the type of road and characteristics about the incident
    ingested_data['Road_Type'] = Feature.create_road_type_column(ingested_data, config['feature_creation'])
    ingested_data[list(config['feature_creation']['incident_type'].keys())] = Feature.created_incident_type_column(
        ingested_data,
        config['feature_creation']
    )
    ## 2.3. Time-of-day and day-of-week metrics.
    ingested_data['Hour'] = ingested_data['Start_Time'].dt.hour # Exatrct hour from start date
    ingested_data['Weekend'] = (ingested_data['Start_Time'].dt.weekday >= 5).astype(np.int32) # Saturday=5, Sunday=6
    ## 2.3.1. This clustering was used for exploratory analysis of any trends. Do certain times of day have different
    #          profiles of severity to others?
    hour_plot, hour_mappings, ingested_data['Hour_Cluster'] = Feature.created_clustered_column(
        ingested_data, # Create a dendrogram based on the severity profile of hour of days. This then forms 4 clusters
        'Hour',
        config['feature_creation']
    )
    ## 2.3.2. This clustering was used to create intuitive time-of-day features, such as morning peak and early morning.
    ingested_data[list(config['feature_creation']['time_of_day'].keys())] = Feature.created_hour_type_column(
        ingested_data,
        config['feature_creation']
    )
    ## 2.4. Weather was clustered to reduce the dimensionality, clustering by the severity profile.
    weather_plot, weather_mappings, ingested_data['Weather_Cluster'] = Feature.created_clustered_column(
        ingested_data,
        'Weather_Condition',
        config['feature_creation']
    )
    ## 2.5. County was clustered to reduce the dimensionality, clustering by the severity profile.
    county_plot, county_mappings, ingested_data['County_Cluster'] = Feature.created_clustered_column(
        ingested_data,
        'County',
        config['feature_creation']
    )
    ## 2.6. Regional council was added as a feature to group counties.
    ingested_data["Regional_Council"] = Feature.create_regional_council_column(ingested_data,config['feature_creation'])
    road_type_plot, road_type_mappings, ingested_data['Road_Type_Cluster'] = Feature.created_clustered_column(
        ingested_data,
        'Road_Type',
        config['feature_creation']
    )
    ## 2.7. Severity was defined as a binary problem as whether it was greater than 2 or not.
    ingested_data['Severity_Gt_2'] = (ingested_data['Severity']>2).astype(np.int32)
    ## 2.8. The descriptive statistics for this enriched data were created
    Desctiptive.generate_descriptives(ingested_data, definitions).to_csv("results/descriptives/ingested.csv")
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
    :return: Outputs the modelling results.
    """
    print("Starting Modelling")
    # Open config with parameters for the modelling process.
    with open("conf.yaml", 'r') as config_file:
        # Read config file
        config = yaml.safe_load(config_file)
    # Step 1: The enriched features are read back in
    ingested_data = pd.read_csv("results/preprocessing/ingested_data.csv")
    ingested_data = Ingestion.parse_datetime_columns(ingested_data, config['ingestion'])

    # Step 2: Get dummies and perform any required imputation. Then subset columns and split into test / train
    enriched_data = Modelling.impute_missing_values(ingested_data, config['modelling'])
    enriched_data = pd.get_dummies(enriched_data, columns=config['modelling']['columns_to_dummy'], drop_first=True)

    enriched_data = enriched_data[[config['modelling']['response'], "Start_Time", *config['modelling']['predictors']]]

    # ## 3.1. Remove observations after June 2020, upon which it appears the classification of severity changes systemically.
    # enriched_data[~((enriched_data["Start_Time"].dt.month >= 6) & (enriched_data["Start_Time"].dt.year == 2020))]
    # Step 4: Split into train and validation, where the validation is solely used for assessing performance of model.
    train_data, validation_data = train_test_split(enriched_data, test_size=0.2, random_state=42)

    # train_data = enriched_data[~((enriched_data["Start_Time"].dt.month>=8) & (enriched_data["Start_Time"].dt.year==2020))]
    # validation_data = enriched_data[
    #     (enriched_data["Start_Time"].dt.month >= 8) & (enriched_data["Start_Time"].dt.year == 2020)
    # ]


    # Step 4: Fit Regularised Linear Regression model
    train_data_X = train_data[config['modelling']['predictors']]
    train_data_Y = train_data[config['modelling']['response']]
    validation_data_X = validation_data[config['modelling']['predictors']]
    validation_data_Y = validation_data[config['modelling']['response']]
    model_error_values, optimal_alpha, model, normaliser = Modelling.fit_cv_model(
        train_data_X,
        train_data_Y,
        chosen_alpha=110
    )

    # Step 5: Calculate diagnostic metrics
    ## 5.1. The train and validation error should be roughly equal if overfitting is negligible
    print('Train AUC value is {:.2f} and validation AUC value is {:.2f}'.format(
        metrics.roc_auc_score(
            train_data_Y,
            model.predict_proba(normaliser.transform(train_data_X))[:,1]
        ),
        metrics.roc_auc_score(
            validation_data_Y,
            model.predict_proba(normaliser.transform(validation_data_X))[:,1]
        ),
    ))
    ## 5.2. The confusion matrix describes the predicted vs observed severity using a likelihood split of 0.5
    conf_mat = confusion_matrix(validation_data_Y, model.predict(normaliser.transform(validation_data_X)))
    ## 5.3. The CV plot shows the error at different levels of regularisation.
    cv_fig = Modelling.compute_cv_curve(model_error_values, optimal_alpha, chosen_alpha=110)
    ## 5.4. The coefficients describe the predictors selected by LASSO and their impact on the fit.
    coefficients = pd.DataFrame(zip(train_data_X.columns, model.coef_[0]), columns=["predict", "coef_estimate"])
    ## 5.5. The AUC plot
    auc_fig = Modelling.create_auc_plot(model, validation_data_X, validation_data_Y)
    # Step n: Save results to the results folder
    if not os.path.exists("results/modelling"):
        os.makedirs("results/modelling")
    coefficients.to_csv("results/modelling/coef_estimates.csv") # Write out the coefficients
    cv_fig.savefig("results/modelling/cv_fig.png", bbox_inches="tight") # Write out the CV plot for the LASSO fitting
    auc_fig.savefig("results/modelling/auc_fig.png", bbox_inches="tight")  # Write out the AUC plot
    pd.DataFrame( # Write out a confision matrix
        conf_mat,
        columns=[f'Predicted {x}' for x in range(0,2)],
        index=[f'Observed {x}' for x in range(0,2)]
    ).to_csv("results/modelling/conf_mat.csv")


if __name__ == "__main__":
    # These two parts of the pipeline are split into two, since each is time-consuming.
    # ingestion_and_enrichment()
    modelling_and_diagnostics()
