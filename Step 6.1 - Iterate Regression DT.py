#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:46:41 2023

@author: leemingjun
"""

import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


######## Dataset Dictionary for Training and Preditions ########

# File path
folder_path_TP_DT = "/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Treated Datasets/Decision Tree/"

# List of dataset file names
DT_datasets = ['DT_MCAR_1K_25.csv','DT_MCAR_1K_25_50.csv', 'DT_MCAR_1K_25_75.csv', 'DT_MCAR_1K_50_25.csv', 'DT_MCAR_1K_50_50.csv', 'DT_MCAR_1K_50_75.csv',
            'DT_MCAR_1K_75_25.csv', 'DT_MCAR_1K_75_50.csv', 'DT_MCAR_1K_75_75.csv',
            'DT_MCAR_10K_25.csv','DT_MCAR_10K_25_50.csv', 'DT_MCAR_10K_25_75.csv', 'DT_MCAR_10K_50_25.csv', 'DT_MCAR_10K_50_50.csv', 'DT_MCAR_10K_50_75.csv',
            'DT_MCAR_10K_75_25.csv', 'DT_MCAR_10K_75_50.csv', 'DT_MCAR_10K_75_75.csv',
            'DT_MCAR_100K_25.csv','DT_MCAR_100K_25_50.csv', 'DT_MCAR_100K_25_75.csv', 'DT_MCAR_100K_50_25.csv', 'DT_MCAR_100K_50_50.csv', 'DT_MCAR_100K_50_75.csv',
            'DT_MCAR_100K_75_25.csv', 'DT_MCAR_100K_75_50.csv', 'DT_MCAR_100K_75_75.csv',
            'DT_MNAR_1K_25.csv', 'DT_MNAR_1K_25_50.csv', 'DT_MNAR_1K_25_75.csv', 'DT_MNAR_1K_50_25.csv', 'DT_MNAR_1K_50_50.csv', 'DT_MNAR_1K_50_75.csv',
            'DT_MNAR_1K_75_25.csv', 'DT_MNAR_1K_75_50.csv', 'DT_MNAR_1K_75_75.csv',
            'DT_MNAR_10K_25.csv', 'DT_MNAR_10K_25_50.csv', 'DT_MNAR_10K_25_75.csv', 'DT_MNAR_10K_50_25.csv', 'DT_MNAR_10K_50_50.csv', 'DT_MNAR_10K_50_75.csv',
            'DT_MNAR_10K_75_25.csv', 'DT_MNAR_10K_75_50.csv', 'DT_MNAR_10K_75_75.csv',
            'DT_MNAR_100K_25.csv', 'DT_MNAR_100K_25_50.csv', 'DT_MNAR_100K_25_75.csv', 'DT_MNAR_100K_50_25.csv', 'DT_MNAR_100K_50_50.csv', 'DT_MNAR_100K_50_75.csv',
            'DT_MNAR_100K_75_25.csv', 'DT_MNAR_100K_75_50.csv', 'DT_MNAR_100K_75_75.csv'
           ]

# Dictionary to store the datasets
DT_imputed_dataset = {}

for dataset_file in DT_datasets:
    # Construct the full file path
    file_path = os.path.join(folder_path_TP_DT, dataset_file)
    
    # Load the dataset
    dataset = pd.read_csv(file_path)
    
    # Store the dataset in the dictionary with the file name as the key
    DT_imputed_dataset[dataset_file] = dataset

    

######## Undergo Regression Prediction Modelling ########


######## For Prediction Modelling ########

# Specify the target column 'N_Var3'
target_col = 'N_Var3'


######## Iterate the Regression Modelling ########

DT_regression_predicted = {}
y_test_regression_DT = {}

# Iterate over the datasets in DT_imputed_dataset
for treated_dataset_name, treated_dataset in DT_imputed_dataset.items():

    # Train the prediction model
    model_info, y_test_df= regression_train_prediction_model(treated_dataset, target_col, treated_dataset_name)
    
    y_test_regression_DT[treated_dataset_name] = y_test_df

    # Make predictions on the training dataset
    data_predicted = regression_make_predictions(treated_dataset, model_info, target_col)
    
    DT_regression_predicted[treated_dataset_name] = data_predicted
    

######## Caluculate Performance Metrics ########

Performance_REG_DF_DT = pd.DataFrame(columns=["Predicted Dataset", "True Dataset", "Mean Squared Error", "Rooted Mean Square Error",
                                                 "Mean Absolute Error", "R-Squared"])

for data_predicted_name, data_predicted in DT_regression_predicted.items():
    
    # Select the true dataset based on the number of rows in the predicted dataset
    if data_predicted_name in  y_test_regression_DT:
        true_dataset_name = data_predicted_name

    # Load the true values for the target variable
    true_values = y_test_regression_DT[true_dataset_name][target_col]
    
    mse, rmse, mae, r2 = calculate_regression_metrics(true_values, data_predicted, target_col)
    
    # Create a temporary DataFrame with the current iteration results
    temporary_reg_df = pd.DataFrame({"Predicted Dataset": [data_predicted_name],
                                 "True Dataset": [true_dataset_name],
                                 "Mean Squared Error": [mse],
                                 "Rooted Mean Square Error": [rmse],
                                 "Mean Absolute Error": [mae],
                                 "R-Squared": [r2]})

    # Concatenate the temporary DataFrame with the Performance_REG_DF_DT DataFrame
    Performance_REG_DF_DT = pd.concat([Performance_REG_DF_DT, temporary_reg_df], ignore_index=True)
        
    
#Performance_REG_DF_DT.to_csv('/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Results/Regression/Performance_REG_DT.csv')

