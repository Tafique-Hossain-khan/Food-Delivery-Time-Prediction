from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.feature_extraction import FeatureExtraction
from src.entity.config_entity import (
                                TrainingPipelineConfig,
                                DataTransformationConfig,
                                DataValidationConfig,
                                DataIngestionConfig,
                                ModelTrainerConfig,
                                FeatureExtractionConfig,
                                MissingValueImputerConfig
                                )
from src.components.data_transformation import DataTransformation
from src.exception.exception import CustomeException
from src.components.handle_missing_value import MissingValueImputer
from src.components.model_trainer import ModelTrainer

from src.entity.artifact_entity import MissingValueImputerArtifact




import sys
if __name__ == "__main__":
    try:
        trainingpipelineconfig=TrainingPipelineConfig()

        #Data ingestion
        dataIngestionConfig = DataIngestionConfig(training_pipeline_config=trainingpipelineconfig)
        obj = DataIngestion(dataIngestionConfig)
        dataIngestionaartifact = obj.initiate_data_ingestion()

        #data validation
        #dataValidationConfig = DataValidationConfig(training_pipeline_config=trainingpipelineconfig)
        #dataValidationObj = DataValidation(data_ingestion_artifact=dataIngestionaartifact,data_validation_config=dataValidationConfig)
        #dataValidationArtifacts = dataValidationObj.initiate_data_validation()

        #handleing missing value
        
        missingValueImputerConfig = MissingValueImputerConfig(training_pipeline_config=trainingpipelineconfig)
        missingValueImputerObj = MissingValueImputer(missing_value_imputer_config=missingValueImputerConfig,data_ingestion_artifact=dataIngestionaartifact,strategy='Fill')
        missingValueImputerArtifact = missingValueImputerObj.initiate_missing_value_imputer()


        #feature extraction
        featureEngineeringConfig = FeatureExtractionConfig(training_pipeline_config=trainingpipelineconfig)
        featureEngineeringObj = FeatureExtraction(clean_data_artifact=missingValueImputerArtifact,feature_extraction_config=featureEngineeringConfig)
        featureEngineeringArtifact = featureEngineeringObj.initiate_feature_extraction()


        #data_transformation
        dataTransformationConfig = DataTransformationConfig(training_pipeline_config=trainingpipelineconfig)
        dataTransformationObj = DataTransformation(feature_extraction_artifact=featureEngineeringArtifact,data_transformation_config=dataTransformationConfig)
        dataTransformationArtifact = dataTransformationObj.initiate_data_transformation()

        #Model Trainer
        modelTrainerConfig = ModelTrainerConfig(training_pipeline_config=trainingpipelineconfig)
        modelTrainerObj = ModelTrainer(data_transformation_artifact=dataTransformationArtifact,model_trainer_config=ModelTrainerConfig)
        modelTrainerObj.initiate_model_trainer()
    
    except Exception as e:
        raise CustomeException(e,sys)