"""
This module orchestrates the data ingestion pipeline for the diabetes prediction project.

It performs the following tasks:

1. Reads the raw diabetes dataset from "notebook/data/diabetes.csv".
2. Splits the data into training and testing sets.
3. Saves the split data to "artifacts/".

"""

import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion process.

    Attributes:
        train_data_path: Path to store the training data (str).
        test_data_path: Path to store the testing data (str).
        raw_data_path: Path to the raw diabetes data (str).
    """

    train_data_path: str=os.path.join("artifacts", "train.csv")
    test_data_path: str=os.path.join("artifacts", "test.csv")
    raw_data_path: str=os.path.join("artifacts", "data.csv")

class DataIngestion:
    """
    Class responsible for ingesting and preparing the diabetes prediction dataset.

    Attributes:
        ingestion_config: Configuration object for data ingestion (DataIngestionConfig).
    """

    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the raw data, splits it, and saves the training and testing sets.

        Raises:
            CustomException: If any error occurs during processing.

        Returns:
            Tuple[str, str]: Paths to the saved training and testing data.
        """

        logging.info("Entered the data ingestion")
        try:
            df = pd.read_csv('notebook\\data\\diabetes.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=2)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Initiate data ingestion
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    # Perform data transformation
    data_transformation=DataTransformation()
    train_arr,test_arr,file_path=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))