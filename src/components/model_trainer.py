import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import customException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class modelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class modelTrainer:
    def __init__(self):
        self.modelTrainerConfig=modelTrainerConfig()

    def initiateModelTraining(self, train_array,test_array):
        try:
            logging.info('Splitting training and testing input data')
            xtrain,ytrain,xtest,ytest=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'LinearRegression':LinearRegression(),
                'K-Neighbours Regressor': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor()
            }  

            model_report:dict=evaluate_model(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest,models=models) 
            
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise customException("No best model found")

            logging.info('Best model found on both training and testing dataset')

            save_object(
                file_path=self.modelTrainerConfig.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(xtest)

            r2_square=r2_score(ytest,predicted)
            return r2_square
            

        except Exception as e:
            raise customException(e,sys)


