from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.pipelines.training_pipeline import train_pipeline
from src.components.model_trainer import ModelTrainer

ing = DataIngestion()
train_path, test_path = ing.initiate_data_ingestion()

trans = DataTransformation()
trans.initiate_data_transformation(train_path, test_path)

train_arr,test_arr=train_pipeline.start_training_pipeline()

train_model=ModelTrainer()
train_model.initate_model_training(train_arr,test_arr)
