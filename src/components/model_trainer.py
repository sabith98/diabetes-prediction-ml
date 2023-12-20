"""
This module trains and selects the best model for the diabetes prediction project.
It handles:
- Defining multiple machine learning models with hyperparameter grids.
- Evaluating model performance.
- Selecting the best-performing model based on test scores.
- Detecting overfitting and raising exceptions if necessary.
- Saving the best model to a file for later use.
"""

import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    """Configuration for model training."""
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    """Class responsible for model training and selection."""

    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        """
        Trains, evaluates, and selects the best model.

        Args:
            train_array: NumPy array containing training data.
            test_array: NumPy array containing testing data.

        Returns:
            tuple[str, dict, float]:
                - Name of the best-performing model.
                - Best hyperparameters for the selected model.
                - Accuracy score of the best model on the testing set.

        Raises:
            CustomException: If no model with sufficient performance is found or if overfitting is detected.
        """

        try:
            logging.info("Split training and test input data")
            x_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            # Define models with hyperparameter grids
            models = {
                'Logistic Regression': (LogisticRegression(random_state=2), {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100]
                }),
                'Random Forest': (RandomForestClassifier(random_state=2), {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30]
                }),
                'SVM': (SVC(random_state=2), {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }),
                'KNN': (KNeighborsClassifier(), {
                    'n_neighbors': [3, 5, 7, 9],
                    'p': [1, 2]
                }),
                'Decision Tree': (DecisionTreeClassifier(random_state=2), {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }),
                'Gradient Boosting': (GradientBoostingClassifier(random_state=2), {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.1, 0.2]
                }),
                'Naive Bayes': (GaussianNB(), {}),
            }

            best_models:dict=evaluate_models(X_train=x_train,y_train=y_train,
                                             X_test=X_test,y_test=y_test,models=models)
            
            # Select and evaluate the best model
            best_model_name = self._select_best_model(best_models)
            best_model = best_models[best_model_name]['model']

            # Assess overfitting
            self._check_overfitting(best_models, best_model_name)
         
            logging.info("Best found model on both training and testing dataset")
            
            # Save and return best model results
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy_score_value = accuracy_score(y_test, predicted)

            return (
                best_model_name,
                best_models[best_model_name]["best_params"],
                accuracy_score_value
            )

        except Exception as e:
            raise CustomException(e,sys)
        
    def _select_best_model(self, best_models: dict) -> str:
        """Selects the best model based on test score."""

        best_model_name = max(best_models, key=lambda k: best_models[k]["test_score"])
        if best_models[best_model_name]["test_score"] < 0.6:
            raise CustomException("No best model found with sufficient performance")
        return best_model_name

    def _check_overfitting(self, best_models: dict, best_model_name: str):
        """Checks for overfitting and raises an exception if necessary."""

        overfitting_difference = best_models[best_model_name]["train_score"] - best_models[best_model_name]["test_score"]
        if overfitting_difference > 0.05:
            raise CustomException(f"{best_model_name} model overfit by {overfitting_difference*100}%")