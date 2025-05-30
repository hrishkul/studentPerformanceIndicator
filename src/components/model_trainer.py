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
                'Decision Tree': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor()
            }  

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "RandomForestRegressor": {   
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor": {  
                    'learning_rate': [.1,.01,.05,.001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression": {},   
                "XGBRegressor": {
                    'learning_rate': [.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor": {    
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor": {   
                    'learning_rate': [.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report:dict=evaluate_model(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest,models=models,params=params) 
            
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


