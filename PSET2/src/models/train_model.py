import json
import yaml
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import warnings
import logging
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config
    
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    
    print("RMSE: %s" %rmse)
    print("MAE: %s" %mae)
    return rmse, mae

def get_feat_and_target(df,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    x=df.drop(target,axis=1)
    y=df[[target]]
    return x,y 
    
def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]
    alpha = config["lasso_config"]["alpha"]
    
    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x,train_y=get_feat_and_target(train,target)
    test_x,test_y=get_feat_and_target(test,target)
    
    imputer = SimpleImputer(strategy="median")
    
    train_x_imputed=imputer.fit_transform(train_x)
    
    scaler = StandardScaler()
    train_x_scl = scaler.fit_transform(train_x_imputed)
    #train_y_values=train_y.values()
    
    test_x_imputed=imputer.transform(test_x)
    test_x_scl=scaler.transform(test_x_imputed)
    #test_y_values=test_y.values()
    
    mlflow_config = config["mlflow_config"]
    
    mlflow.set_tracking_uri("https://dagshub.com/AnweshaMohanty-07/MLOps-Project.mlflow")
    tracking_uri = mlflow.get_tracking_uri()
    print("Current tracking uri: {}".format(tracking_uri))

    mlflow.set_experiment(mlflow_config["experiment_name"])
    
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)
    
    with mlflow.start_run():
        lin_reg = Lasso(alpha=alpha)
        lin_reg.fit(train_x_scl,train_y)
        
        y_predict=lin_reg.predict(test_x_scl)
        (rmse,mae)=eval_metrics(test_y,y_predict)
        
        mlflow.log_param("alha",alpha)
        mlflow.log_metric("RMSE",rmse)
        mlflow.log_metric("MAE",mae)
        
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lin_reg, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(model, "lin_reg")
    
if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args=args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
    
    