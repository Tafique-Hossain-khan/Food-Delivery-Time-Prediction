import pandas as pd
import os,sys
from abc import ABC,abstractmethod
from src.logger import logging
from src.exception import CustomException

# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file."""
        pass

class CsvDataIngestion(DataIngestor):
    
    def ingest(self, file_path: str) -> pd.DataFrame:

        try:
            # ensure that correct file is given
            if not file_path.endswith(".csv"):
                raise ValueError("The provided file is not in .csv format ")
            
            data = pd.read_csv(file_path)
            logging.info(data.shape)
            return data
        except Exception as e:
            raise CustomException(e,sys)


class DataIngestorFactory:

    @staticmethod
    def get_data_ingestor(file_extension:str):

        if file_extension == ".csv":
            return CsvDataIngestion()
        else:
            raise ValueError("Not a valid format")

            
'''
if __name__=="__main__":

    logging.info("Staring data ingestion ")
    file_path = 'data/train.csv'
    file_extension = ".csv"
    csv_ingester = DataIngestorFactory.get_data_ingestor(file_extension)
    df = csv_ingester.ingest(file_path)
    logging.info(df.shape)
'''