import os
import sys
from src.exception import customException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import dataTransform
from src.components.data_transformation import dataTransformConfig
from src.components.model_trainer import modelTrainerConfig
from src.components.model_trainer import modelTrainer

@dataclass
class dataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'raw.csv')

class dataIngestion:
    def __init__(self):
        self.ingestion_config=dataIngestionConfig()

    def initiateDI(self):
        logging.info("Entered the Data Ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info('Read the dataset as DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train-Test split initiated')
            trainSet,testSet=train_test_split(df,test_size=0.2,random_state=42)
            trainSet.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            testSet.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion has been completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
        except Exception as e:
            raise customException(e,sys)
        

if __name__=='__main__':
    obj=dataIngestion()
    train_data, test_data=obj.initiateDI()
    data_transform=dataTransform()
    train_arr,test_arr,_=data_transform.initiateDataTransform(train_data,test_data)

    modetrainer=modelTrainer()
    print(modetrainer.initiateModelTraining(train_arr,test_arr))