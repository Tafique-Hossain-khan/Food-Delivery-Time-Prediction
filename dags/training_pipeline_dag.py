from airflow import DAG
from airflow.decorators import task
from datetime import datetime

from src.logging.logger import logging
from src.exception.exception import CustomeException
from src.entity.config_entity import TrainingPipelineConfig
from src.pipeline.training_pipeline import TrainingPipeline


from airflow.decorators import dag, task
from datetime import datetime



training_pipeline_config = TrainingPipelineConfig()

# Initialize TrainingPipeline
pipeline = TrainingPipeline(training_pipeline_config)

@dag(
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["training_pipeline"],
)
def training_pipeline_dag():
    
    @task
    def data_ingestion_task():
        return pipeline.start_data_ingestion()

    @task
    def missing_value_imputer_task(data_ingestion_artifact):
        return pipeline.start_missing_value_imputer(data_ingestion_artifact)

    @task
    def feature_extraction_task(missing_value_imputer_artifact):
        return pipeline.start_feature_extraction(missing_value_imputer_artifact)

    @task
    def data_transformation_task(feature_extraction_artifact):
        return pipeline.start_data_transformation(feature_extraction_artifact)

    @task
    def model_trainer_task(data_transformation_artifact):
        return pipeline.start_model_trainer(data_transformation_artifact)

    # Define task dependencies
    data_ingestion = data_ingestion_task()
    missing_value_imputer = missing_value_imputer_task(data_ingestion)
    feature_extraction = feature_extraction_task(missing_value_imputer)
    data_transformation = data_transformation_task(feature_extraction)
    model_trainer = model_trainer_task(data_transformation)

training_pipeline_dag = training_pipeline_dag()
