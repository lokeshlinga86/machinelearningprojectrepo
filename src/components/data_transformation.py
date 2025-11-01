import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            numerical_features=["reading_score","writing_score"]
            categorical_features=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            num_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy='median')),("scaler",StandardScaler(with_mean=False))])
            cat_pipelines=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),("one_hot_encoder",OneHotEncoder()),("scaler",StandardScaler(with_mean=False))])
            logging.info("categorical columns encoding completed")
            logging.info("numerical columns encoding completed ")
            preproccessor=ColumnTransformer([("numpipeline",num_pipeline,numerical_features),("cat_pipelines",cat_pipelines,categorical_features)])
            return preproccessor

        except  Exception as e:
            raise CustomException(e,sys)
        




        
    def initate_data_transformation(self,train_path,test_path):
            try:
                train_df=pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)
                logging.info("read train and test data completed")
                logging.info("obtaining preprocessing object")
                preproccessing_obj=self.get_data_transformer_object()
                target_column_name="math_score"
                numerical_columns=["writing_score","reading_score"]
                input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
                target_feature_train_df=train_df[target_column_name]
                input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
                target_feature_test_df=test_df[target_column_name]
                logging.info("applying preprocessing object on train and test datadataframe")
                input_feature_train_arr=preproccessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr=preproccessing_obj.transform(input_feature_test_df)
                train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
                test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
                
                logging.info(f"saved preprocessing project")
                save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preproccessing_obj)
                return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
            except Exception as e:
                raise CustomException(e,sys)

