#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 23:22:52 2023

@author: leemingjun
"""

# Generating MNAR missing values

# 25, 25

def introduce_mnar_25(dataset, column_indices, missing_percentage):
    # Convert sys_sample_1K to a numpy array
    dataset_array = dataset.values

    # Calculate the number of missing values based on the percentage
    num_rows, num_columns = dataset.shape
    num_missing = min(int((missing_percentage / 100) * num_rows), num_rows)

    # Generate the row indices for every 4th row
    row_indices_5 = np.arange(0, num_rows, 4)

    # Set missing values at the specific rows and columns
    for col_index in column_indices:
        # Generate patterned indices for missing values in the current column
        if col_index == 5:
            missing_indices = np.random.choice(row_indices_5, size=num_missing, replace=False)
        elif col_index == 7:
            row_indices_7 = np.arange(0, num_rows, 4)
            missing_indices = np.random.choice(row_indices_7, size=num_missing, replace=False)

        # Assign nan to missing indices for the selected row and column
        dataset_array[missing_indices, col_index] = np.nan

    return dataset_array


# Generating 50% MNAR

def introduce_mnar_50(dataset, column_indices, missing_percentage):
    # Convert sys_sample_1K to a numpy array
    dataset_array = dataset.copy().values

    # Identify the number of rows  and columns
    num_rows, num_columns = dataset_array.shape

    # Generate the row indices for every 2nd row
    row_indices = np.arange(0, num_rows, 2)

    for col_index, missing_percentage in zip(column_indices, missing_percentage):
        num_missing_existing = mnar_1K_25_df_copy.iloc[:, col_index].isnull().sum()
        num_missing_total = int((missing_percentage / 100) * num_rows)
        num_missing_to_add = num_missing_total - num_missing_existing

        # Set missing values at the specific rows and columns
        if num_missing_to_add > 0:
            # Generate patterned indices for missing values in the current column
            missing_indices = np.random.choice(row_indices, size=num_missing_to_add, replace=False)

            # Assign nan to missing indices for the selected row and column
            dataset_array[missing_indices, col_index] = np.nan
        
            # Check the number of missing values after the initial addition
            num_missing_added = pd.isnull(dataset_array[:, col_index]).sum()
            
            num_missing_count = int((missing_percentage / 100) * num_rows)
    
            # If the number is less than 500, continue generating additional random indices until 500
            while num_missing_added < num_missing_count:
                remaining_to_add = num_missing_count - num_missing_added
                additional_indices = np.random.choice(row_indices, size=remaining_to_add, replace=False)

                # Assign additional nan to missing indices for the selected row and column
                dataset_array[additional_indices, col_index] = np.nan

                # Update the count of missing values
                num_missing_added = pd.isnull(dataset_array[:, col_index]).sum()

    return dataset_array
    

# Generating 75% MNAR

def introduce_mnar_75(dataset, column_indices, missing_percentage):
    # Convert sys_sample_1K to a numpy array
    dataset_array = dataset.copy().values

    # Identify the number of rows  and columns
    num_rows, num_columns = dataset_array.shape

    for col_index, missing_percent in zip(column_indices, missing_percentage):
        num_missing_existing = mnar_1K_25_df_copy.iloc[:, col_index].isnull().sum()
        num_missing_total = int((missing_percent / 100) * num_rows)
        num_missing_to_add = num_missing_total - num_missing_existing

        # Determine the pattern parameters
        pattern_length = 4  # 4 rows
        missing_rows = 3    # 3 missing values

        if num_missing_to_add > 0:
            num_patterns = num_rows // pattern_length  # Number of complete patterns (1000/25)
            remaining_rows = num_rows % pattern_length  # Remaining rows after complete patterns, quotient

            # Set missing values for patterns_indices_75
            pattern_indices_75 = np.arange(0, num_patterns * pattern_length, pattern_length)
            for pattern_index in pattern_indices_75:
                missing_indices = np.arange(pattern_index, pattern_index + missing_rows)
                dataset_array[missing_indices, col_index] = np.nan

            # Set missing values for remaining rows (if any)
            if remaining_rows >= missing_rows:
                remaining_pattern_indices = np.arange(num_patterns * pattern_length, num_rows, 1)
                remaining_missing_indices = remaining_pattern_indices[:missing_rows]
                dataset_array[remaining_missing_indices, col_index] = np.nan

    return dataset_array
