#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:58:34 2023

@author: leemingjun
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os

# File path
folder_path = "/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Datasets/"

# List of dataset file names
datasets = ['MCAR_1K_25.csv','MCAR_1K_25_50.csv', 'MCAR_1K_25_75.csv', 'MCAR_1K_50_25.csv', 'MCAR_1K_50_50.csv', 'MCAR_1K_50_75.csv',
            'MCAR_1K_75_25.csv', 'MCAR_1K_75_50.csv', 'MCAR_1K_75_75.csv',
            'MCAR_10K_25.csv','MCAR_10K_25_50.csv', 'MCAR_10K_25_75.csv', 'MCAR_10K_50_25.csv', 'MCAR_10K_50_50.csv', 'MCAR_10K_50_75.csv',
            'MCAR_10K_75_25.csv', 'MCAR_10K_75_50.csv', 'MCAR_10K_75_75.csv',
            'MCAR_100K_25.csv','MCAR_100K_25_50.csv', 'MCAR_100K_25_75.csv', 'MCAR_100K_50_25.csv', 'MCAR_100K_50_50.csv', 'MCAR_100K_50_75.csv',
            'MCAR_100K_75_25.csv', 'MCAR_100K_75_50.csv', 'MCAR_100K_75_75.csv',
            'MNAR_1K_25.csv', 'MNAR_1K_25_50.csv', 'MNAR_1K_25_75.csv', 'MNAR_1K_50_25.csv', 'MNAR_1K_50_50.csv', 'MNAR_1K_50_75.csv',
            'MNAR_1K_75_25.csv', 'MNAR_1K_75_50.csv', 'MNAR_1K_75_75.csv',
            'MNAR_10K_25.csv', 'MNAR_10K_25_50.csv', 'MNAR_10K_25_75.csv', 'MNAR_10K_50_25.csv', 'MNAR_10K_50_50.csv', 'MNAR_10K_50_75.csv',
            'MNAR_10K_75_25.csv', 'MNAR_10K_75_50.csv', 'MNAR_10K_75_75.csv',
            'MNAR_100K_25.csv', 'MNAR_100K_25_50.csv', 'MNAR_100K_25_75.csv', 'MNAR_100K_50_25.csv', 'MNAR_100K_50_50.csv', 'MNAR_100K_50_75.csv',
            'MNAR_100K_75_25.csv', 'MNAR_100K_75_50.csv', 'MNAR_100K_75_75.csv',
           ]

# Dictionary to store the datasets
dataset_dict = {}

for dataset_file in datasets:
    # Construct the full file path
    file_path = os.path.join(folder_path, dataset_file)
    
    # Load the dataset
    dataset = pd.read_csv(file_path)
    
    dataset = dataset.drop(columns="Unnamed: 0")
    
    # Store the dataset in the dictionary with the file name as the key
    dataset_dict[dataset_file] = dataset


######## Undergo Data Treatment: Decision Tree ########

# Dictionary to store the imputed datasets
DT_imputed_datasets = {}

# Iterate over the datasets in the dataset_dict dictionary
for dataset_name, dataset_DT in dataset_dict.items():
    # Extract the missing columns from the dataset
    missing_cols = dataset_DT.columns[dataset_DT.isnull().any()].tolist()
    
    # Apply the impute_missing_values function to the dataset
    imputed_dataset = impute_missing_values(dataset_DT, missing_cols)
    
    # Add the prefix to the dataset name
    output_dataset_name = "DT_" + dataset_name
    
    # Store the imputed dataset in the dictionary
    DT_imputed_datasets[output_dataset_name] = imputed_dataset
    
# =============================================================================
#     # Create a variable with the dataset name and assign the imputed DataFrame
#     locals()[output_dataset_name] = imputed_dataset
# =============================================================================
    
    # Calculate and print the number of missing values for the imputed dataset
    missing_values = imputed_dataset[missing_cols].isnull().sum()
    print("Number of missing values for", output_dataset_name + ":", missing_values)


######## Export the datasets ########


# =============================================================================
# output_directory = '/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Treated Datasets/Decision Tree'
# 
# # Iterate over the datasets in the dataset_dict dictionary
# for dataset_name, dataset in DT_imputed_datasets.items():
#     # Define the full file path including the directory and filename
#     filename = f"{output_directory}/{dataset_name}"
#     
#     # Save the dataset as a CSV file
#     dataset.to_csv(filename, index=False)
#     
#     print(f"Dataset '{dataset_name}' saved as '{filename}'.")
# =============================================================================


######## Discriptive Analysis on Imputed Datasets ########


# =============================================================================
# # Iterate over the datasets in the DT_imputed_datasets dictionary
# for dataset_name, dataset in DT_imputed_datasets.items():
#     print(f"\nDataset: {dataset_name}")
#     
#     # Select the categorical columns 'C_Var2' and 'C_Var4'
#     categorical_cols = dataset[['C_Var2', 'C_Var4']]
#     
#     # Calculate the frequency count and proportion for each categorical column
#     for col in categorical_cols:
#         print(f"\nDescriptive analysis for column: {col}")
#         
#         # Calculate the Frequency count of each unique value in the column
#         frequency_count = dataset[col].value_counts()
#         print(frequency_count)
#         
#         # Calculate the Proportion of each unique value (Unique Count//Total Count)
#         proportion = dataset[col].value_counts(normalize=True)
#         print(proportion)
# =============================================================================
