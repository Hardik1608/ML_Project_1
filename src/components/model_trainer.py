import os 
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import(AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging


from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('Project_Data', 'model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def intiata_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data')
            X_train, X_test, y_train, y_test= (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )

            models = {
                'Random_Forest': RandomForestRegressor(),
                'Decision_Tree': DecisionTreeRegressor(),
                'Gradient_Boosting': GradientBoostingRegressor(),
                'Linear_Regression': LinearRegression(),
                'K-Neighbors_Regressor': KNeighborsRegressor(),
                'XGB_Regressor': XGBRegressor(),
                'CatBoost_Regressor': CatBoostRegressor(),
                'Adaboost_Regressor': AdaBoostRegressor()

            }

            params={
                "Decision_Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random_Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "K-Neighbors_Regressor":{
                    'n_neighbors': [1, 5, 10, 15, 20],
                    'weights': ['uniform', 'distance'],
                    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    # 'leaf_size': [10, 30, 50],
                    'p': [1, 2],
                    # 'metric': ['minkowski', 'manhattan', 'euclidean']

                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    
                },                
                "Gradient_Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear_Regression":{},
                "XGB_Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost_Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "Adaboost_Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            model_report: dict= evaluate_models(X_train=X_train, X_test = X_test, y_train = y_train, y_test = y_test, models = models, params = params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)

            ]
            best_model = models[best_model_name]

            if best_model_score< 0.6:
                raise CustomException('No best model found.')
            
            logging.info('Best model found.')

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted = best_model.predict(X_test)
            score = r2_score(y_test, predicted)

            return (f'{best_model_name} performs the best with an r2 score of {score}.')
        
        except Exception as e:
            raise CustomException(e, sys)