import pandas as pd
from typing import Dict

class Ingestion(object):
    @staticmethod
    def parse_datetime_columns(raw_file: pd.DataFrame, ingestion_config: Dict) -> pd.DataFrame:
        """
        :param raw_file: The raw file prior to any preprocessing
        :param ingestion_config: The ingestion specific config
        :return: A dataframe with all columns converted to datetime
        """
        for column, form in ingestion_config['timestamp_columns'].items():
            raw_file[column] = pd.to_datetime(
                raw_file[column],
                format=form
            )
        return raw_file

    @staticmethod
    def dedupe_asset(raw_file: pd.DataFrame, ingestion_config: Dict) -> pd.DataFrame:
        """
        This dedupes by taking the first record where multiple exist
        :param raw_file: The file with datetime columns converted to timestamps. This is important to ensure the deduping
            is successful
        :param ingestion_config: The config parameters related to ingestion
        :return: The deduped asset
        """
        # There are multiple key combinations used to dedupe. This is because not all dupes are the same.
        raw_data_deduped = raw_file
        for deduping_keys in ingestion_config['deduping_keys']:
            print(f"Previous row size: {len(raw_data_deduped)}")
            print(f"Deduping keys: {str(deduping_keys)}")
            raw_data_deduped.drop_duplicates(subset=deduping_keys, inplace=True)
            print(f"Subsequent row size: {len(raw_data_deduped)}")

        return raw_data_deduped
