"""
This module handles the prediction pipeline for the diabetes prediction project.
It includes:
- Loading the trained model and preprocessor.
- Transforming new input data using the preprocessor.
- Making predictions using the loaded model.
- Handling potential exceptions.
"""

import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    """Class responsible for making predictions using a trained model."""

    def __init__(self):
        pass

    def predict(self,features):
        """
        Makes predictions using the loaded model.

        Args:
            features: Pandas DataFrame containing new input data.

        Returns:
            Pandas Series containing the predicted values (0 or 1).

        Raises:
            CustomException: If any errors occur during prediction.
        """

        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            print(data_scaled)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    """Class to encapsulate input data for prediction."""

    def __init__(  self,
        Pregnancies: int,
        Glucose: int,
        BloodPressure: int,
        SkinThickness: int,
        Insulin: int,
        BMI: float,
        DiabetesPedigreeFunction: float,
        Age: int):

        """
        Initializes a CustomData object with feature values.

        Args:
            Keyword arguments containing feature values.
        """

        self.Pregnancies = Pregnancies
        self.Glucose = Glucose
        self.BloodPressure = BloodPressure
        self.SkinThickness = SkinThickness
        self.Insulin = Insulin
        self.BMI = BMI
        self.DiabetesPedigreeFunction = DiabetesPedigreeFunction
        self.Age = Age

    def get_data_as_data_frame(self):
        """
        Returns the input data as a Pandas DataFrame.

        Returns:
            Pandas DataFrame containing the input data.

        Raises:
            CustomException: If any errors occur during DataFrame creation.
        """
        
        try:
            custom_data_input_dict = {
                "Pregnancies": [self.Pregnancies],
                "Glucose": [self.Glucose],
                "BloodPressure": [self.BloodPressure],
                "SkinThickness": [self.SkinThickness],
                "Insulin": [self.Insulin],
                "BMI": [self.BMI],
                "DiabetesPedigreeFunction": [self.DiabetesPedigreeFunction],
                "Age": [self.Age],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)