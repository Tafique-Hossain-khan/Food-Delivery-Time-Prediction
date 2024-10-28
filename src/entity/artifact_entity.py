from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_file_path:str
    trained_file_path:str
    test_file_path:str


@dataclass
class DataValidationArtifact:
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class MissingValueImputerArtifact:
    cleaned_data_file_path: str
    

@dataclass
class FeatureExtractionArtifact:
    cleaned_data_file_path: str
    

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class RegressionMetricArtifact:
    mean_absolute_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    r2_score: float
    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    