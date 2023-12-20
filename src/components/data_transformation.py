"""
This module implements data transformation for the diabetes prediction project.
It handles:
- Creating a preprocessor object to standardize numerical features.
- Applying the preprocessor to training and testing data.
- Saving the preprocessor object for future use.
"""

import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation process.

    Attributes:
        preprocessor_obj_file_path: Path to save the preprocessor pickle file (str).
    """

    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    """
    Class responsible for transforming data in the diabetes prediction pipeline.

    Attributes:
        data_transformation_config: Configuration object for data transformation (DataTransformationConfig).
    """

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates a preprocessor object.

        Returns:
            ColumnTransformer: The preprocessor object.

        Raises:
            CustomException: If an error occurs during preprocessing.
        """

        try:
            numerical_columns = [
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age",
            ]

            num_pipeline= Pipeline(
                steps=[
                    ("scaler",StandardScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
            
    def initiate_data_transformation(self,train_path,test_path):
        """
        Applies data transformation to training and testing data.

        Args:
            train_path: Path to the training data CSV file.
            test_path: Path to the testing data CSV file.

        Returns:
            tuple[np.ndarray, np.ndarray, str]:
                - Transformed training data as a NumPy array.
                - Transformed testing data as a NumPy array.
                - Path to the saved preprocessor object.

        Raises:
            CustomException: If an error occurs during transformation.
        """

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="Outcome"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr=np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)