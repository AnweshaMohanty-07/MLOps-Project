import numpy as np
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import warnings
import logging
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


mlflow.set_tracking_uri("https://dagshub.com/AnweshaMohanty-07/MLOps-Project.mlflow")
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    mse = np(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    return mse, mae
    
if __name__=="__main__":
    housing=pd.read_csv("MLOps PSET2/MLOPs-Project/PSET2/data/raw/housing.csv")
    housing.drop("ocean_proximity",axis=1,inplace=True)
    housing["income_cat"] = pd.cut(housing["median_income"],bins=[0., 1.5, 3., 4.5, 6., np.inf ], labels=[1,2,3,4,5])

    df_train_strat, df_test_strat = train_test_split(housing,test_size=0.2,random_state=42,stratify=housing["income_cat"])
    df_train_strat.drop("income_cat", axis=1, inplace=True) # income_cat was created for stratified split. Its use is over. Let us remove it.
    df_test_strat.drop("income_cat", axis=1, inplace=True)

    hml = df_train_strat.drop("median_house_value", axis=1) # drop labels for training set
    hml_labels = df_train_strat["median_house_value"].copy() 

    imputer = SimpleImputer(strategy="median")
    imputer.fit(hml)
    X = imputer.transform(hml) # Imputer returns a numpy array. So we need to transform it back to a pandas df

    hml_num_tr = pd.DataFrame(data=X,columns=hml.columns,index=hml.index)

    hml_num_tr["bedrooms_per_room"] = hml_num_tr["total_bedrooms"]/hml_num_tr["total_rooms"]
    hml_num_tr["bedrooms_per_house"] = hml_num_tr["total_bedrooms"]/hml_num_tr["households"]
    hml_num_tr["rooms_per_house"] = hml_num_tr["total_rooms"]/hml_num_tr["households"]

    X_train = hml_num_tr.to_numpy()
    y_train = hml_labels.values

    scaler = StandardScaler()
    X_train_scl = scaler.fit_transform(X_train)



    hml_test = df_test_strat.drop("median_house_value", axis=1) # drop labels for test set
    hml_test_labels = df_test_strat["median_house_value"].copy()

    X_test1=imputer.transform(hml_test)

    hml_num_test = pd.DataFrame(data=X_test1,columns=hml_test.columns,index=hml_test.index)
    hml_num_test["bedrooms_per_room"] = hml_num_test["total_bedrooms"]/hml_num_test["total_rooms"]
    hml_num_test["bedrooms_per_house"] = hml_num_test["total_bedrooms"]/hml_num_test["households"]
    hml_num_test["rooms_per_house"] = hml_num_test["total_rooms"]/hml_num_test["households"]

    X_test = hml_num_test.to_numpy()
    y_test = hml_test_labels.values

    X_test_scl=scaler.transform(X_test)
    
    with mlflow.start_run():
        lin_reg = LinearRegression()
        lin_reg.fit(X_train_scl,y_train)

        y_predict=lin_reg.predict(X_test_scl)
        
        (mse,mae)=eval_metrics(y_test,y_predict)
        
        print("MSE: %s", mse)
        print("MAE: %s",mae)
        
        mlflow.log_mteric("MSE",mse)
        mlflow.log_metric("MAE",mae)
        
        mlflow.sklearn.log_model(lin_reg,"model")






