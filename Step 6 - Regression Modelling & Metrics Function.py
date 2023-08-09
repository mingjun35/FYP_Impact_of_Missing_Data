#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:13:59 2023

@author: leemingjun
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

def regression_train_prediction_model(dataset, target_col, dataset_name):
    
    np.random.seed(42)
    
    if "DELETE" in dataset_name:
        
        # Create a new DataFrame without the target column
        X = dataset.drop(target_col, axis=1)

        X = X.set_index("Dup_Index")

        # Extract the Dup_Index and target variable (Dup_Index is to trace back index)
        y = dataset[["Dup_Index", target_col]]

        # Set Dup_Index as index
        y = y.set_index("Dup_Index")
        
        
        ########### One-hot encoding for categorical columns ###########

        # Identify categorical variables and perform one-hot encoding
        categorical_cols = X.select_dtypes(include='object').columns
        encoder = OneHotEncoder(sparse=False)


        # Fit the encoder on the categorical columns in the training dataset
        encoder.fit(X[categorical_cols])

        X_encoded = pd.DataFrame(encoder.transform(X[categorical_cols]))
        X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
        X_encoded = pd.concat([X.drop(categorical_cols, axis=1).reset_index(drop=False), X_encoded], axis=1)

        # Set Dup_Index as index
        X_encoded = X_encoded.set_index("Dup_Index")
        
        # Get the encoded column names - this for later inverse transform
        encoded_column_names = encoder.get_feature_names_out(categorical_cols)


        ########### Train the Regression model ###########

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

        # Create an instance of the LinearRegression model
        regression = LinearRegression()

        # Fit the model to the training data
        regression.fit(X_train, y_train)
        
        y_test_df = pd.DataFrame(y_test, index=y_test.index)
        y_test_df.columns = [target_col]

        y_test_df = y_test_df.sort_index()

        # Save the trained model and encoder for later prediction
        model_info = {
            'regression': regression,
            'encoder': encoder,
            'categorical_cols': categorical_cols,
            'target_col': target_col,
            'X_test': X_test,
            'encoded_column_names': encoded_column_names
           }
        
        return model_info, y_test_df
    
    
    else:
        
        # Create a new DataFrame without the target column
        X = dataset.drop(target_col, axis=1)
        
        # Extract the target variable
        y = dataset[target_col]
        
        
        ########### One-hot encoding for categorical columns ###########

        # Identify categorical variables and perform one-hot encoding
        categorical_cols = X.select_dtypes(include='object').columns
        encoder = OneHotEncoder(sparse=False)
        
        # Fit the encoder on the categorical columns in the training dataset
        encoder.fit(X[categorical_cols])
        
        X_encoded = pd.DataFrame(encoder.transform(X[categorical_cols]))
        X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
        X_encoded = pd.concat([X.drop(categorical_cols, axis=1).reset_index(drop=True), X_encoded], axis=1)
        
        # Get the encoded column names - this for later inverse transform
        encoded_column_names = encoder.get_feature_names_out(categorical_cols)
        
        
        ########### Train the Regression model ###########

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

        # Create an instance of the LinearRegression model
        regression = LinearRegression()

        # Fit the model to the training data
        regression.fit(X_train, y_train)
        
        y_test_df = pd.DataFrame(y_test, index=y_test.index)
        y_test_df.columns = [target_col]

        y_test_df = y_test_df.sort_index()
        
        # Save the trained model and encoder for later prediction
        model_info = {
            'regression': regression,
            'encoder': encoder,
            'categorical_cols': categorical_cols,
            'target_col': target_col,
            'X_test': X_test,
            'y_test': y_test,
            'encoded_column_names': encoded_column_names
           }
        
        return model_info, y_test_df
        
        
def regression_make_predictions(dataset, model_info, target_col):
    
    np.random.seed(42)
    
    # Retrieve the model information
    regression = model_info['regression']
    encoder = model_info['encoder']
    categorical_cols = model_info['categorical_cols']
    target_col = model_info['target_col']
    X_test = model_info['X_test']
    #y_test = model_info['y_test']
    encoded_column_names = model_info['encoded_column_names']
    
    # Copy the original treated dataset
    data_predicted = X_test.copy()

    # Drop the columns in encoded_column_names
    data_predicted = data_predicted.drop(encoded_column_names, axis=1)

    # Sort the index in ascending order
    data_predicted = data_predicted.sort_index()

    # Make predictions using the trained model
    predictions = regression.predict(X_test)
    
    # Inverse transform X_test
    X_test_categorical_inverse = pd.DataFrame(encoder.inverse_transform(X_test[encoded_column_names]), index=X_test.index)
    X_test_categorical_inverse.columns = categorical_cols

    # Sort the index in ascending order
    X_test_categorical_inverse = X_test_categorical_inverse.sort_index()

    # Replace the predicted column(s) in the data_predicted with the predictions
    # Merge the inverse transformed categorical columns back into data_predicted
    data_predicted[categorical_cols] = X_test_categorical_inverse
    data_predicted[target_col] = predictions

    # Rearrange the columns in the desired order
    desired_order = ["N_Var1", "N_Var2", "N_Var3", "N_Var4", "C_Var1", "C_Var2", "C_Var3", "C_Var4"]
    data_predicted = data_predicted[desired_order]
    
    return data_predicted


def calculate_regression_metrics(true_values, data_predicted, target_col):
    
    true_values, data_predicted = true_values.align(data_predicted, axis=0, copy=False)
    
    predictions = data_predicted["N_Var3"]
    
    mse = mean_squared_error(true_values, predictions)
    
    rmse = mean_squared_error(true_values, predictions, squared=False)
    
    mae = mean_absolute_error(true_values, predictions)
    
    r2 = r2_score(true_values, predictions)
    
    return mse, rmse, mae, r2
    
    
