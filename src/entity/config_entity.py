from datetime import datetime
import os



class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.artifact_name="Artifacts"
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        self.model_dir=os.path.join("final_model")
        self.timestamp: str=timestamp

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.raw_data_path:str = os.path.join("Data/ingested_data","raw_data.csv")
        self.train_data_path:str = os.path.join("Data/ingested_data","train_data.csv")
        self.test_data_path:str = os.path.join("Data/ingested_data","test_data.csv")

class MissingValueImputerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.cleaned_data_path:str = os.path.join(training_pipeline_config.artifact_dir,"cleaned_data","cleaned_data.csv")


class DataValidationConfig: 
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        
        self.drift_report_file_path = os.path.join(training_pipeline_config.artifact_dir,"Drift_report","report.yaml")
        self.valid_train_data_path = os.path.join(training_pipeline_config.artifact_dir,"valid_data","train.csv")
        self.valid_test_data_path = os.path.join(training_pipeline_config.artifact_dir,"valid_data","test.csv")
        self.invalid_train_data_path = os.path.join(training_pipeline_config.artifact_dir,"invalid_data","train.csv")
        self.invalid_test_data_path = os.path.join(training_pipeline_config.artifact_dir,"invalid_data","test.csv")

class FeatureExtractionConfig:
    
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.extracted_data_path = os.path.join("Data/Extracted_Features","new_data.csv")

        

class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.transformed_train_file_path = os.path.join(training_pipeline_config.artifact_dir,"data_transformation","transformed_data","train.npy")
        self.transformed_test_file_path = os.path.join(training_pipeline_config.artifact_dir,"data_transformation","transformed_data","test.npy")
        self.transformed_object_file_path =  os.path.join(training_pipeline_config.artifact_dir,"data_transformation","transformed_object","preprocessor.pkl")


class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.final_model_file_path = os.path.join(training_pipeline_config.artifact_dir,"model_trainer","trained_model","model.pkl")

        