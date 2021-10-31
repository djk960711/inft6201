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
        deduping_keys = ingestion_config['deduping_keys']
        raw_data_deduped = pd.concat([
            rows_for_key.sort_index().head(1)
            for _, rows_for_key
            in raw_file.groupby(by=deduping_keys)
        ])
        return raw_data_deduped
