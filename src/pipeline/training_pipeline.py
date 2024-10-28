
import os
import sys

from src.exception.exception import CustomeException
from src.logging.logger import logging

from src.components.data_ingestion import DataIngestion
#from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.handle_missing_value import MissingValueImputer
from src.components.feature_extraction import FeatureExtraction

from src.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    FeatureExtractionArtifact,
    MissingValueImputerArtifact
)
from src.entity.config_entity import MissingValueImputerConfig,FeatureExtractionConfig
import sys



class TrainingPipeline:

    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
        
        

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data Ingestion")
            data_ingestion=DataIngestion(data_ingestion_config_obj=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomeException(e,sys)
    '''
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_validation_config=DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=data_validation_config)
            logging.info("Initiate the data Validation")
            data_validation_artifact=data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise CustomeException(e,sys)
    '''
    def start_missing_value_imputer(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            missing_value_imputer_conifg = MissingValueImputerConfig(training_pipeline_config=self.training_pipeline_config)
            missing_value_imputer_obj = MissingValueImputer(missing_value_imputer_config=missing_value_imputer_conifg,data_ingestion_artifact=data_ingestion_artifact,strategy='Fill')
            missing_valueImputer_artifact = missing_value_imputer_obj.initiate_missing_value_imputer()

            return missing_valueImputer_artifact
        
        except Exception as e:
            raise CustomeException(e,sys)
    def start_feature_extraction(self,missing_value_imputer_artifact:MissingValueImputerArtifact):
        try:
            feature_extraction_config = FeatureExtractionConfig(training_pipeline_config=self.training_pipeline_config)
            feature_extraction_obj = FeatureExtraction(clean_data_artifact=missing_value_imputer_artifact,feature_extraction_config=feature_extraction_config)
            feature_extraction_artifact = feature_extraction_obj.initiate_feature_extraction()

            return feature_extraction_artifact
        except Exception as e:
            raise CustomeException(e,sys)
    
    
    def start_data_transformation(self,feature_extraction_artifact:FeatureExtractionArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation_obj = DataTransformation(feature_extraction_artifact=feature_extraction_artifact,data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation_obj.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise CustomeException(e,sys)
        
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact

        except Exception as e:
            raise CustomeException(e, sys)

    
    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            missing_value_imputer_artifact = self.start_missing_value_imputer(data_ingestion_artifact)
            feature_extraction_artifact = self.start_feature_extraction(missing_value_imputer_artifact)
            data_transformation_artifact = self.start_data_transformation(feature_extraction_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            
    
            
            return model_trainer_artifact
        except Exception as e:
            raise CustomeException(e,sys)
        
    

if __name__ == "__main__":
    obj = TrainingPipeline()
    obj.run_pipeline()
