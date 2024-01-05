from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Perform data ingestion
obj = DataIngestion()
train_data,test_data=obj.initiate_data_ingestion()

# Perform data transformation
data_transformation=DataTransformation()
train_arr,test_arr,file_path=data_transformation.initiate_data_transformation(train_data,test_data)

# Perform model training
modeltrainer=ModelTrainer()
print(modeltrainer.initiate_model_trainer(train_arr,test_arr))