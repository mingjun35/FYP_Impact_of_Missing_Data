#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 23:00:11 2023

@author: leemingjun
"""

# Generating MCAR missing values


# 25, 25

def introduce_mcar_25(dataset, column_indices, missing_percentage):
    # Convert sys_sample_1K to a numpy array
    dataset_array = dataset.values

    # Calculate the number of missing values based on the percentage
    num_rows, num_columns = dataset.shape
    num_missing = min(int((missing_percentage / 100) * num_rows), num_rows)

    # Iterate over the column indices
    for col_index in column_indices:
        # Generate random indices for missing values in the current column
        missing_indices = np.random.choice(num_rows, size=num_missing, replace=False)
        
        # Assign nan to missing indices to the current column
        dataset_array[missing_indices, col_index] = np.nan

    return dataset_array


# 25, 50


def introduce_mcar_50(dataset, column_indices, missing_percentages):
    # Convert mcar_1K_25_df to a numpy array
    dataset_array = dataset.copy().values

    # Identify the number of rows  and columns
    num_rows, num_columns = dataset_array.shape

    for col_index, missing_percentage in zip(column_indices, missing_percentages):
        # Calculate the number of missing values to add for the current column
        num_missing_existing = dataset.iloc[:, col_index].isnull().sum()
        num_missing_total = int((missing_percentage / 100) * num_rows)
        num_missing_to_add = num_missing_total - num_missing_existing
        
        if num_missing_to_add > 0:
            # Generate random indices for missing values in the current column
            missing_indices = np.random.choice(num_rows, size=num_missing_to_add, replace=False)

            # Assign nan to missing indices to the current column
            dataset_array[missing_indices, col_index] = np.nan
            
            # Check the number of missing values after the initial addition
            num_missing_added = pd.isnull(dataset_array[:, col_index]).sum()
            
            num_missing_count = int((missing_percentage / 100) * num_rows)

            # If the number is less than 500, continue generating additional random indices until it reaches 500
            while num_missing_added < num_missing_count:
                remaining_to_add = num_missing_count - num_missing_added
                additional_indices = np.random.choice(num_rows, size=remaining_to_add, replace=False)

                # Set the selected column indices to nan at the additional missing indices
                dataset_array[additional_indices, col_index] = np.nan

                # Update the count of missing values
                num_missing_added = pd.isnull(dataset_array[:, col_index]).sum()

    return dataset_array


# 25, 75


def introduce_mcar_75(dataset, column_indices, missing_percentages):
    # Convert mcar_1K_25_50_df to a numpy array
    dataset_array = dataset.copy().values

    # Identify the number of rows  and columns
    num_rows, num_columns = dataset_array.shape

    for col_index, missing_percentage in zip(column_indices, missing_percentages):
        # Calculate the number of missing values to add for the current column
        num_missing_existing = dataset.iloc[:, col_index].isnull().sum()
        num_missing_total = int((missing_percentage / 100) * num_rows)
        num_missing_to_add = num_missing_total - num_missing_existing
        
        if num_missing_to_add > 0:
            # Generate random indices for missing values in the current column
            missing_indices = np.random.choice(num_rows, size=num_missing_to_add, replace=False)

            # Assign nan to missing indices to the current column
            dataset_array[missing_indices, col_index] = np.nan
            
            # Check the number of missing values after the initial addition
            num_missing_added = pd.isnull(dataset_array[:, col_index]).sum()
            
            num_missing_count = int((missing_percentage / 100) * num_rows)

            # If the number is less than 750, continue generating additional random indices until 750
            while num_missing_added < num_missing_count:
                remaining_to_add = num_missing_count - num_missing_added
                additional_indices = np.random.choice(num_rows, size=remaining_to_add, replace=False)

                # Set the selected column indices to nan at the additional missing indices
                dataset_array[additional_indices, col_index] = np.nan

                # Update the count of missing values
                num_missing_added = pd.isnull(dataset_array[:, col_index]).sum()

    return dataset_array

