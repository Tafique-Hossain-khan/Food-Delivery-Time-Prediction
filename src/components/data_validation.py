from src.logging.logger import logging
from src.exception.exception import CustomeException
from src.utils.main_utlis import utils
from scipy.stats import ks_2samp # used to compare the distribution of two sampel
import os,sys,yaml
import pandas as pd
from src.entity.config_entity import TrainingPipelineConfig,DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact
from src.utils.main_utlis.utils import read_yaml
from src.entity.artifact_entity import DataIngestionArtifact


class DataValidation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        
        try:
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise CustomeException(e,sys)
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame,yaml_file_path:str)->bool:
        """
        This fucntion aims to validate the number of columns in the dataset

        Parameters:
        dataframe: The original dataset
        yaml_file_path: File path for the schema of the dataset

        Return: 
        Boolean value 'True' if the columns are same else 'False'
        """
        try:
            schema = utils.read_yaml(yaml_file_path)
            schema = schema["columns"]
            number_of_columns=len(schema)
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        except Exception as e:
            raise CustomeException(e,sys)
        
    def validate_number_of_numerical_columsn(self,dataframe:pd.DataFrame,yaml_file_path):
        """
        This fucntion aims to validate the number of numerical columns in the dataset

        Parameters:
        dataframe: The original dataset
        yaml_file_path: File path for the schema of the dataset

        Return: 
        Boolean value 'True' if the columns are same else 'False'
        """
        pass

    

    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        """
        This fucntion aims check the drift in the upcomming dataset

        Parameters:
        base_df (pd.DataFrame): Original Dataset that is being used to train the model
        current_df (pd.DataFrame): new Dtatset 

        Return: 
        Boolean value 'True' if the columns are same else 'False'
        """
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue: #if the p_value is greater than the thresold then there is no drift
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
                
                os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path),exist_ok=True)
                utils.write_yaml_file(file_path=self.data_validation_config.drift_report_file_path,content=report)

        except Exception as e:
            raise CustomeException(e,sys)
        
    def initiate_data_validation(self):
        try:
            
            train_dataframe = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_dataframe = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            #train_dataframe = pd.read_csv("Data\ingested_data\\train_data.csv")
            #test_dataframe = pd.read_csv("Data\ingested_data\\test_data.csv")
            yaml_file_path = 'Data_schema\schema.yaml'
            columns_status_values = []
            status=self.validate_number_of_columns(dataframe=train_dataframe,yaml_file_path=yaml_file_path)
            columns_status_values.append(status)
            
            if not status:
                error_message=f"Train dataframe does not contain all columns.\n"
            status = self.validate_number_of_columns(dataframe=test_dataframe,yaml_file_path=yaml_file_path)
            columns_status_values.append(status)
            
            if not status:
                error_message=f"Test dataframe does not contain all columns.\n"   
            ## lets check datadrift
            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)

            #Based on the status we can save the incomming data
            drift_report_content = read_yaml(self.data_validation_config.drift_report_file_path)
            #with open(self.data_validation_config.drift_report_file_path ,'r') as file:
            #   y = yaml.safe_load(file)
            drift_report = []
            for flag in drift_report_content.values():
                drift_report.append(list(flag.values())[0])

            drift_report_status = all(value is False for value in drift_report)
            columns_status_values = all(value is False for value in columns_status_values)

            logging.info(columns_status_values)
            logging.info(drift_report_content)
            
            print(drift_report_status)
            print(columns_status_values)
            if columns_status_values==True and drift_report_status==True:
                os.makedirs(os.path.dirname(self.data_validation_config.valid_train_data_path),exist_ok=True)
                train_dataframe.to_csv(self.data_validation_config.valid_train_data_path)

                os.makedirs(os.path.dirname(self.data_validation_config.valid_test_data_path),exist_ok=True)
                test_dataframe.to_csv(self.data_validation_config.valid_test_data_path)

            else:
                os.makedirs(os.path.dirname(self.data_validation_config.invalid_train_data_path),exist_ok=True)
                train_dataframe.to_csv(self.data_validation_config.invalid_train_data_path)

                os.makedirs(os.path.dirname(self.data_validation_config.invalid_test_data_path),exist_ok=True)
                test_dataframe.to_csv(self.data_validation_config.invalid_test_data_path)


            data_validation_artifacts = DataValidationArtifact(
                
                valid_train_file_path = self.data_validation_config.valid_train_data_path,
                valid_test_file_path = self.data_validation_config.valid_test_data_path,
                invalid_train_file_path = self.data_validation_config.invalid_train_data_path,
                invalid_test_file_path = self.data_validation_config.invalid_test_data_path,
                drift_report_file_path = self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifacts
        except Exception as e:
            raise CustomeException(e,sys)

