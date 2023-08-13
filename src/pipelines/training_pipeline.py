import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion

from src.components.data_transformation import DataTransformation
from dataclasses import dataclass
## from src.components.model_trainer import ModelTrainer

class train_pipeline:

       def start_training_pipeline():
           obj=DataIngestion()
           train_data_path,test_data_path=obj.initiate_data_ingestion()
           print(train_data_path,test_data_path)

           data_transformation=DataTransformation()

           train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

           return train_arr,test_arr
       