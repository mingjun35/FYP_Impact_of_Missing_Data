#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:24:52 2023

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
    
    

######## Undergo Decision Tree Prediction Modelling ########


######## For Prediction Modelling ########

# Specify the target column 'C_Var3'
target_col = 'C_Var3'


######## Iterate the Prediction Modelling ########

DT_DT_predicted = {}
y_test_dict_DT = {}

# Iterate over the datasets in DT_imputed_dataset
for treated_dataset_name, treated_dataset in DT_imputed_dataset.items():

    # Train the prediction model
    model_info, y_test_categorical_inverse = train_prediction_model(treated_dataset, target_col, treated_dataset_name)
    
    y_test_dict_DT[treated_dataset_name] = y_test_categorical_inverse

    # Make predictions on the training dataset
    data_predicted = make_predictions(treated_dataset, model_info, target_col)
    
    DT_DT_predicted[treated_dataset_name] = data_predicted
    

######## Evaluation Metrics ########
    

######## Caluculate Misclassification Rate ########

# Create an empty DataFrame to store the results
Misclasification_DF_DT = pd.DataFrame(columns=["Predicted Dataset", "True Dataset", "Misclassification Rate"])

for data_predicted_name, data_predicted in DT_DT_predicted.items():
    
    # Select the true dataset based on the number of rows in the predicted dataset
    if data_predicted_name in  y_test_dict_DT:
        true_dataset_name = data_predicted_name

    # Load the true values for the target variable
    true_values = y_test_dict_DT[true_dataset_name][target_col]    
    
    # Calculate the misclassification rate
    misclassification_rate = calculate_misclassification_rate(true_values, data_predicted, data_predicted_name)

    # Format the misclassification rate with two decimal places
    misclassification_rate_formatted = "{:.2%}".format(misclassification_rate)
    
    # Print the dataset name and misclassification rate
    #print("{} - {}, MisClass Rate: {:.2%}".format(treated_dataset_name, true_dataset_name, float(misclassification_rate)))

    # Create a temporary DataFrame with the current iteration results
    temp_df = pd.DataFrame({"Predicted Dataset": [data_predicted_name],
                            "True Dataset": [true_dataset_name],
                            "Misclassification Rate": [misclassification_rate_formatted]})

    # Concatenate the temporary DataFrame with the Misclasification_DF_DT DataFrame
    Misclasification_DF_DT = pd.concat([Misclasification_DF_DT, temp_df], ignore_index=True) 
    
print(Misclasification_DF_DT)


#Misclasification_DF_DT.to_csv('/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Results/Decision Tree/Misclassification_DT_DT.csv')


######## Caluculate Accuracy ########

# Create an empty DataFrame to store the results
Accuracy_DF_DT = pd.DataFrame(columns=["Predicted Dataset", "True Dataset", "Accuracy"])

for data_predicted_name, data_predicted in DT_DT_predicted.items():
    
    # Select the true dataset based on the number of rows in the predicted dataset
    if data_predicted_name in  y_test_dict_DT:
        true_dataset_name = data_predicted_name

    # Load the true values for the target variable
    true_values = y_test_dict_DT[true_dataset_name][target_col]  
    
    # Calculate the accuracy
    accuracy = calculate_accuracy(true_values, data_predicted, data_predicted_name)
    
    # Format the misclassification rate with two decimal places
    accuracy_formatted = "{:.2%}".format(accuracy)
    
    # Create a temporary DataFrame with the current iteration results
    temp_df = pd.DataFrame({"Predicted Dataset": [data_predicted_name],
                            "True Dataset": [true_dataset_name],
                            "Accuracy": [accuracy_formatted]})
    
    Accuracy_DF_DT = pd.concat([Accuracy_DF_DT, temp_df], ignore_index=True) 

print(Accuracy_DF_DT)


#Accuracy_DF_DT.to_csv('/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Results/Decision Tree/Accuracy_DT_DT.csv')


######## Tabulate Confusion Matrix ########

Confusion_Matrix_DF_DT = pd.DataFrame(columns=["Predicted Dataset", "True Dataset", "True Positive", "False Positive", 
                                                "False Negative", "True Negative"])

for data_predicted_name, data_predicted in DT_DT_predicted.items():
    
    # Select the true dataset based on the number of rows in the predicted dataset
    if data_predicted_name in  y_test_dict_DT:
        true_dataset_name = data_predicted_name

    # Load the true values for the target variable
    true_values = y_test_dict_DT[true_dataset_name][target_col]  
    
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

    # Concatenate the temporary DataFrame with the Confusion_Matrix_DF_DT DataFrame
    Confusion_Matrix_DF_DT = pd.concat([Confusion_Matrix_DF_DT, temp_df], ignore_index=True)
    
print(Confusion_Matrix_DF_DT)


######## Calculate f1-score, precision, sensitivity, specificity ########

Performance_DF_DT = pd.DataFrame(columns=["Predicted Dataset", "True Dataset", "Precision", 
                                            "Sensitivity (Recall +)", "Specificity (Recall -)", "F1-Score"])

for index, row in Confusion_Matrix_DF_DT.iterrows():
    predicted_datasets = row["Predicted Dataset"]
    true_datasets = row["True Dataset"]  
    true_positive = row["True Positive"]
    false_positive = row["False Positive"]
    false_negative = row["False Negative"]
    true_negative = row["True Negative"]

    precision, sensitivity, specificity, f1_score = calculate_metrics(Confusion_Matrix_DF_DT)

    # Create a temporary DataFrame with the current iteration results
    temporary_df = pd.DataFrame({"Predicted Dataset": [predicted_datasets],
                                 "True Dataset": [true_datasets],
                                 "Precision": [precision],
                                 "Sensitivity (Recall +)": [sensitivity],
                                 "Specificity (Recall -)": [specificity],
                                 "F1-Score": [f1_score]})

    # Concatenate the temporary DataFrame with the Performance_DF_DT DataFrame
    Performance_DF_DT = pd.concat([Performance_DF_DT, temporary_df], ignore_index=True)


#Performance_DF_DT.to_csv('/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Results/Decision Tree/Performance_DT_DT.csv')






######## For Checking Purpose Only ########


# =============================================================================
#     ######## Oberserve Misclassification Instances ########
#     
#     
#     # Call the function to get the necessary data
#     misclassified_indices, misclassified_true_values, misclassified_predicted_values = observe_misclassified_instances(true_values, data_predicted, target_col)
#     
#     # Create a DataFrame from the misclassified data
#     Misclassified_df = pd.DataFrame({
#         "Index": misclassified_indices,
#         "True Value": misclassified_true_values,
#         "Predicted Value": misclassified_predicted_values
#         })
#     
#     # Add the prefix to the dataset name
#     output_dataset_name = "True_" + treated_dataset_name
#     
#     # Store the imputed dataset in the dictionary
#     Observe_instances[output_dataset_name] = Misclassified_df
# =============================================================================


#print(Observe_instances)    

# =============================================================================
# output_directory = '/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Oberserve Misclassified Instances/True vs DT/'
# 
# # Iterate over the datasets in the dataset_dict dictionary
# for observe_name, observe in Observe_instances.items():
#     # Define the full file path including the directory and filename
#     filename = f"{output_directory}/{observe_name}"
#     
#     # Save the dataset as a CSV file
#     observe.to_csv(filename, index=False)
#     
#     print(f"Dataset '{observe_name}' saved as '{filename}'.")
# =============================================================================

