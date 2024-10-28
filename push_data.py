import os
import sys
from dotenv import load_dotenv
import certifi # use to make secure http connection
import json
import pymongo
import pandas as pd
import pymongo.mongo_client
from src.exception.exception import CustomeException
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataExtraction:
    """
    A class for extracting data from various sources and processing it.

    This class provides methods to convert CSV data to JSON format 
    and to insert the JSON data into a MongoDB database.

    """
    def __init__(self):
        pass

    def csv_to_json(self,df) ->pd.DataFrame:
        """
        This fucntion will aims to conver the csv data into json object

        Parameters:
        df: pandas DataFrame

        Return:
        json_obj: It will return json object
        """
        try:
            json_str = df.to_json(orient='records')
            json_obj = json.loads(json_str)

            return json_obj
        
        except Exception as e:
            raise CustomeException(e,sys)
        
    def insert_data_mangodb(self,records,database,collection):
        """
        This function aims to push the json object into mongodb database

        Parameters:
        records(json_obj): json objects that need to be inserted into database
        database (str): Name of the database where the data is to be inserted
        collection (str): Name of the collection where the data need to be inserted
        
        """
        try:

            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            db = mongo_client[database]
            Collection = db[collection]
            if isinstance(records, list):
                Collection.insert_many(records)  
            else:
                Collection.insert_one(records)
            return len(records)
        except Exception as e:
            raise CustomeException(e,sys)
    


if __name__ == "__main__":
    obj = DataExtraction()
    df = pd.read_csv("Data\phisingData.csv")
    df.reset_index(drop=True,inplace=True)
    json_obj = obj.csv_to_json(df)

    database = "TAFIQUE"
    Collection = "NetworkData"
    no_of_records = obj.insert_data_mangodb(records=json_obj,database=database,collection=Collection)
    print(no_of_records)