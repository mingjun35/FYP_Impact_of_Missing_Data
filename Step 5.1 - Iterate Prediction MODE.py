#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:57:31 2023

@author: leemingjun
"""

import numpy as np
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


######## Dataset Dictionary for Training and Preditions ########

# File path
folder_path_TP_MODE = "/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Treated Datasets/Mode Imputation/"

# List of dataset file names
MODE_datasets = ['MODE_MCAR_1K_25.csv','MODE_MCAR_1K_25_50.csv', 'MODE_MCAR_1K_25_75.csv', 'MODE_MCAR_1K_50_25.csv', 'MODE_MCAR_1K_50_50.csv', 'MODE_MCAR_1K_50_75.csv',
            'MODE_MCAR_1K_75_25.csv', 'MODE_MCAR_1K_75_50.csv', 'MODE_MCAR_1K_75_75.csv',
            'MODE_MCAR_10K_25.csv','MODE_MCAR_10K_25_50.csv', 'MODE_MCAR_10K_25_75.csv', 'MODE_MCAR_10K_50_25.csv', 'MODE_MCAR_10K_50_50.csv', 'MODE_MCAR_10K_50_75.csv',
            'MODE_MCAR_10K_75_25.csv', 'MODE_MCAR_10K_75_50.csv', 'MODE_MCAR_10K_75_75.csv',
            'MODE_MCAR_100K_25.csv','MODE_MCAR_100K_25_50.csv', 'MODE_MCAR_100K_25_75.csv', 'MODE_MCAR_100K_50_25.csv', 'MODE_MCAR_100K_50_50.csv', 'MODE_MCAR_100K_50_75.csv',
            'MODE_MCAR_100K_75_25.csv', 'MODE_MCAR_100K_75_50.csv', 'MODE_MCAR_100K_75_75.csv',
            'MODE_MNAR_1K_25.csv', 'MODE_MNAR_1K_25_50.csv', 'MODE_MNAR_1K_25_75.csv', 'MODE_MNAR_1K_50_25.csv', 'MODE_MNAR_1K_50_50.csv', 'MODE_MNAR_1K_50_75.csv',
            'MODE_MNAR_1K_75_25.csv', 'MODE_MNAR_1K_75_50.csv', 'MODE_MNAR_1K_75_75.csv',
            'MODE_MNAR_10K_25.csv', 'MODE_MNAR_10K_25_50.csv', 'MODE_MNAR_10K_25_75.csv', 'MODE_MNAR_10K_50_25.csv', 'MODE_MNAR_10K_50_50.csv', 'MODE_MNAR_10K_50_75.csv',
            'MODE_MNAR_10K_75_25.csv', 'MODE_MNAR_10K_75_50.csv', 'MODE_MNAR_10K_75_75.csv',
            'MODE_MNAR_100K_25.csv', 'MODE_MNAR_100K_25_50.csv', 'MODE_MNAR_100K_25_75.csv', 'MODE_MNAR_100K_50_25.csv', 'MODE_MNAR_100K_50_50.csv', 'MODE_MNAR_100K_50_75.csv',
            'MODE_MNAR_100K_75_25.csv', 'MODE_MNAR_100K_75_50.csv', 'MODE_MNAR_100K_75_75.csv'
           ]

# Dictionary to store the datasets
MODE_imputed_dataset = {}

for dataset_file in MODE_datasets:
    # Construct the full file path
    file_path = os.path.join(folder_path_TP_MODE, dataset_file)
    
    # Load the dataset
    dataset = pd.read_csv(file_path)
    
    # Store the dataset in the dictionary with the file name as the key
    MODE_imputed_dataset[dataset_file] = dataset

    

######## Undergo Decision Tree Prediction Modelling ########


######## For Prediction Modelling ########

# Specify the target column 'C_Var3'
target_col = 'C_Var3'


######## Iterate the Prediction Modelling ########

MODE_DT_predicted = {}
y_test_dict_MODE = {}

# Iterate over the datasets in MODE_imputed_dataset
for treated_dataset_name, treated_dataset in MODE_imputed_dataset.items():

    # Train the prediction model
    model_info, y_test_categorical_inverse = train_prediction_model(treated_dataset, target_col, treated_dataset_name)
    
    y_test_dict_MODE[treated_dataset_name] = y_test_categorical_inverse

    # Make predictions on the training dataset
    data_predicted = make_predictions(treated_dataset, model_info, target_col)
    
    MODE_DT_predicted[treated_dataset_name] = data_predicted

  
  
######## Evaluation Metrics ########


######## Caluculate Misclassification Rate ########

# Create an empty DataFrame to store the results
Misclassification_DF_MODE = pd.DataFrame(columns=["Treated Dataset", "True Dataset", "Misclassification Rate"])

for data_predicted_name, data_predicted in MODE_DT_predicted.items():

    # Select the true dataset based on the number of rows in the predicted dataset
    if data_predicted_name in  y_test_dict_MODE:
        true_dataset_name = data_predicted_name

    # Load the true values for the target variable
    true_values = y_test_dict_MODE[true_dataset_name][target_col]

    # Calculate the misclassification rate
    misclassification_rate = calculate_misclassification_rate(true_values, data_predicted, data_predicted_name)

    # Format the misclassification rate with two decimal places
    misclassification_rate_formatted = "{:.2%}".format(misclassification_rate)
    
    # Print the dataset name and misclassification rate
    #print("{} - {}, MisClass Rate: {:.2%}".format(treated_dataset_name, true_dataset_name, float(misclassification_rate)))

    # Create a temporary DataFrame with the current iteration results
    temp_df = pd.DataFrame({"Treated Dataset": [data_predicted_name],
                            "True Dataset": [true_dataset_name],
                            "Misclassification Rate": [misclassification_rate_formatted]})

    # Concatenate the temporary DataFrame with the Misclassification_DF_MODE DataFrame
    Misclassification_DF_MODE = pd.concat([Misclassification_DF_MODE, temp_df], ignore_index=True) 
    
print(Misclassification_DF_MODE)


#Misclassification_DF_MODE.to_csv('/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Results/Decision Tree/Misclassification_DT_MODE.csv')


######## Caluculate Accuracy ########

# Create an empty DataFrame to store the results
Accuracy_DF_MODE = pd.DataFrame(columns=["Predicted Dataset", "True Dataset", "Accuracy"])

for data_predicted_name, data_predicted in MODE_DT_predicted.items():
    
    # Select the true dataset based on the number of rows in the predicted dataset
    if data_predicted_name in  y_test_dict_MODE:
        true_dataset_name = data_predicted_name

    # Load the true values for the target variable
    true_values = y_test_dict_MODE[true_dataset_name][target_col]
    
    # Calculate the misclassification rate
    accuracy = calculate_accuracy(true_values, data_predicted, data_predicted_name)
    
    # Format the misclassification rate with two decimal places
    accuracy_formatted = "{:.2%}".format(accuracy)
    
    # Create a temporary DataFrame with the current iteration results
    temp_df = pd.DataFrame({"Predicted Dataset": [data_predicted_name],
                            "True Dataset": [true_dataset_name],
                            "Accuracy": [accuracy_formatted]})
    
    Accuracy_DF_MODE = pd.concat([Accuracy_DF_MODE, temp_df], ignore_index=True) 

print(Accuracy_DF_MODE)


#Accuracy_DF_MODE.to_csv('/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Results/Decision Tree/Accuracy_DT_MODE.csv')


######## Tabulate Confusion Matrix ########

Confusion_Matrix_DF_MODE = pd.DataFrame(columns=["Predicted Dataset", "True Dataset", "True Positive", "False Positive", 
                                                "False Negative", "True Negative"])

for data_predicted_name, data_predicted in MODE_DT_predicted.items():
    
    # Select the true dataset based on the number of rows in the predicted dataset
    if data_predicted_name in  y_test_dict_MODE:
        true_dataset_name = data_predicted_name

    # Load the true values for the target variable
    true_values = y_test_dict_MODE[true_dataset_name][target_col]
    
    # Obtain confusion matrix
    cm = get_confusion_matrix(true_values, data_predicted, data_predicted_name)
    
    # Extract true positive, false positive, false negative, and true negative values from the confusion matrix
    true_positive = cm[1, 1]
    false_positive = cm[0, 1]
    false_negative = cm[1, 0]
    true_negative = cm[0, 0]
    
    # Create a temporary DataFrame with the current iteration results
    temp_df = pd.DataFrame({"Predicted Dataset": [data_predicted_name],
                            "True Dataset": [true_dataset_name],
                            "True Positive": [true_positive],
                            "False Positive": [false_positive],
                            "False Negative": [false_negative],
                            "True Negative": [true_negative]})

    # Concatenate the temporary DataFrame with the Confusion_Matrix_DF_MODE DataFrame
    Confusion_Matrix_DF_MODE = pd.concat([Confusion_Matrix_DF_MODE, temp_df], ignore_index=True)
    
print(Confusion_Matrix_DF_MODE)


######## Calculate f1-score, precision, sensitivity, specificity ########

Performance_DF_MODE = pd.DataFrame(columns=["Predicted Dataset", "True Dataset", "Precision", 
                                            "Sensitivity (Recall +)", "Specificity (Recall -)", "F1-Score"])

for index, row in Confusion_Matrix_DF_MODE.iterrows():
    predicted_datasets = row["Predicted Dataset"]
    true_datasets = row["True Dataset"]  
    true_positive = row["True Positive"]
    false_positive = row["False Positive"]
    false_negative = row["False Negative"]
    true_negative = row["True Negative"]

    precision, sensitivity, specificity, f1_score = calculate_metrics(Confusion_Matrix_DF_MODE)

    # Create a temporary DataFrame with the current iteration results
    temporary_df = pd.DataFrame({"Predicted Dataset": [predicted_datasets],
                                 "True Dataset": [true_datasets],
                                 "Precision": [precision],
                                 "Sensitivity (Recall +)": [sensitivity],
                                 "Specificity (Recall -)": [specificity],
                                 "F1-Score": [f1_score]})

    # Concatenate the temporary DataFrame with the Performance_DF_MODE DataFrame
    Performance_DF_MODE = pd.concat([Performance_DF_MODE, temporary_df], ignore_index=True)


#Performance_DF_MODE.to_csv('/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Results/Decision Tree/Performance_DT_MODE.csv')

