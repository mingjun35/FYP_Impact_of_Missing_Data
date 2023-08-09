#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 21:51:43 2023

@author: leemingjun
"""

# For loop to undergo missing values generation for MNAR

# Define the list of datasets
datasets_25 = [sys_sample_1K, sys_sample_10K, complete_data]  # Update with your dataset names

# For loop for each dataset to introduce 25% MNAR for all datasets
column_indices = [5, 7]
missing_percentage = 25

for dataset in datasets_25:

    if dataset.shape[0] == 1000:
        mnar_1K_25 = introduce_mnar_25(dataset, column_indices, missing_percentage)
        df_mnar_1K_25 = pd.DataFrame(mnar_1K_25, columns=dataset.columns)
        
        mnar_1K_nan_5_25 = df_mnar_1K_25[df_mnar_1K_25.columns[5]].isnull().sum()
        mnar_1K_nan_7_25 = df_mnar_1K_25[df_mnar_1K_25.columns[7]].isnull().sum()

        print("Number of NaN values 5 in dataset:", mnar_1K_nan_5_25)
        print("Number of NaN values 7 in dataset:", mnar_1K_nan_7_25)
        
    elif dataset.shape[0] == 10000:
        mnar_10K_25 = introduce_mnar_25(dataset, column_indices, missing_percentage)
        df_mnar_10K_25 = pd.DataFrame(mnar_10K_25, columns=dataset.columns)
        
        mnar_10K_nan_5_25 = df_mnar_10K_25[df_mnar_10K_25.columns[5]].isnull().sum()
        mnar_10K_nan_7_25 = df_mnar_10K_25[df_mnar_10K_25.columns[7]].isnull().sum()

        print("Number of NaN values 5 in dataset:", mnar_10K_nan_5_25)
        print("Number of NaN values 7 in dataset:", mnar_10K_nan_7_25)
        
    elif dataset.shape[0] == 100000:
        mnar_100K_25 = introduce_mnar_25(dataset, column_indices, missing_percentage)
        df_mnar_100K_25 = pd.DataFrame(mnar_100K_25, columns=dataset.columns)
        
        mnar_100K_nan_5_25 = df_mnar_100K_25[df_mnar_100K_25.columns[5]].isnull().sum()
        mnar_100K_nan_7_25 = df_mnar_100K_25[df_mnar_100K_25.columns[7]].isnull().sum()

        print("Number of NaN values 5 in dataset:", mnar_100K_nan_5_25)
        print("Number of NaN values 7 in dataset:", mnar_100K_nan_7_25)

# For loop for each dataset to introduce 50% & 75% MNAR for TV2
datasets_25_50 = [df_mnar_1K_25, df_mnar_10K_25, df_mnar_100K_25]

column_indices = [7]
missing_percentages = [50, [75, 0, 0, 0]]

for dataset in datasets_25_50:
    if dataset.shape[0] == 1000:
        # Create a copy of mnar_1K_25_df
        mnar_1K_25_df_copy = df_mnar_1K_25.copy()
        mnar_1K_25_50 = introduce_mnar_50(dataset, column_indices, missing_percentages)
        df_mnar_1K_25_50 = pd.DataFrame(mnar_1K_25_50, columns=dataset.columns)

        mnar_1K_nan_5_25_50 = df_mnar_1K_25_50[df_mnar_1K_25_50.columns[5]].isnull().sum()
        mnar_1K_nan_7_25_50 = df_mnar_1K_25_50[df_mnar_1K_25_50.columns[7]].isnull().sum()

        print("Number of NaN values 5 in dataset:", mnar_1K_nan_5_25_50)
        print("Number of NaN values 7 in dataset:", mnar_1K_nan_7_25_50)
            
    elif dataset.shape[0] == 10000:
        # Create a copy of mnar_1K_25_df
        mnar_10K_25_df_copy = df_mnar_10K_25.copy()
        mnar_10K_25_50 = introduce_mnar_50(dataset, column_indices, missing_percentages)
        df_mnar_10K_25_50 = pd.DataFrame(mnar_10K_25_50, columns=dataset.columns)
            
        mnar_10K_nan_5_25_50 = df_mnar_10K_25_50[df_mnar_10K_25_50.columns[5]].isnull().sum()
        mnar_10K_nan_7_25_50 = df_mnar_10K_25_50[df_mnar_10K_25_50.columns[7]].isnull().sum()

        print("Number of NaN values 5 in dataset:", mnar_10K_nan_5_25_50)
        print("Number of NaN values 7 in dataset:", mnar_10K_nan_7_25_50)

    elif dataset.shape[0] == 100000:
        # Create a copy of mnar_1K_25_df
        mnar_100K_25_df_copy = df_mnar_100K_25.copy()
        mnar_100K_25_50 = introduce_mnar_50(dataset, column_indices, missing_percentages)
        df_mnar_100K_25_50 = pd.DataFrame(mnar_100K_25_50, columns=dataset.columns)
            
        mnar_100K_nan_5_25_50 = df_mnar_100K_25_50[df_mnar_100K_25_50.columns[5]].isnull().sum()
        mnar_100K_nan_7_25_50 = df_mnar_100K_25_50[df_mnar_100K_25_50.columns[7]].isnull().sum()

        print("Number of NaN values 5 in dataset:", mnar_100K_nan_5_25_50)
        print("Number of NaN values 7 in dataset:", mnar_100K_nan_7_25_50)

datasets_25_75 = [df_mnar_1K_25_50, df_mnar_10K_25_50, df_mnar_100K_25_50]

for dataset in datasets_25_75:
    if dataset.shape[0] == 1000:
        # Create a copy of mnar_1K_25_df
        mnar_1K_25_50_df_copy = df_mnar_1K_25_50.copy()
        mnar_1K_25_75 = introduce_mnar_75(dataset, column_indices, missing_percentages)
        df_mnar_1K_25_75 = pd.DataFrame(mnar_1K_25_75, columns=dataset.columns)
            
        mnar_1K_nan_5_25_75 = df_mnar_1K_25_75[df_mnar_1K_25_75.columns[5]].isnull().sum()
        mnar_1K_nan_7_25_75 = df_mnar_1K_25_75[df_mnar_1K_25_75.columns[7]].isnull().sum()

        print("Number of NaN values 5 in dataset:", mnar_1K_nan_5_25_75)
        print("Number of NaN values 7 in dataset:", mnar_1K_nan_7_25_75)
            
    elif dataset.shape[0] == 10000:
        # Create a copy of mnar_1K_25_df
        mnar_10K_25_50_df_copy = df_mnar_10K_25_50.copy()
        mnar_10K_25_75 = introduce_mnar_75(dataset, column_indices, missing_percentages)
        df_mnar_10K_25_75 = pd.DataFrame(mnar_10K_25_75, columns=dataset.columns)
            
        mnar_10K_nan_5_25_75 = df_mnar_10K_25_75[df_mnar_10K_25_75.columns[5]].isnull().sum()
        mnar_10K_nan_7_25_75 = df_mnar_10K_25_75[df_mnar_10K_25_75.columns[7]].isnull().sum()

        print("Number of NaN values 5 in dataset:", mnar_10K_nan_5_25_75)
        print("Number of NaN values 7 in dataset:", mnar_10K_nan_7_25_75)
            
    elif dataset.shape[0] == 100000:
        # Create a copy of mnar_1K_25_df
        mnar_100K_25_50_df_copy = df_mnar_100K_25_50.copy()
        mnar_100K_25_75 = introduce_mnar_75(dataset, column_indices, missing_percentages)
        df_mnar_100K_25_75 = pd.DataFrame(mnar_100K_25_75, columns=dataset.columns)
        
        mnar_100K_nan_5_25_75 = df_mnar_100K_25_75[df_mnar_100K_25_75.columns[5]].isnull().sum()
        mnar_100K_nan_7_25_75 = df_mnar_100K_25_75[df_mnar_100K_25_75.columns[7]].isnull().sum()

        print("Number of NaN values 5 in dataset:", mnar_100K_nan_5_25_75)
        print("Number of NaN values 7 in dataset:", mnar_100K_nan_7_25_75)


# For loop for each dataset to introduce 50% MNAR for TV1
datasets_50 = [df_mnar_1K_25, df_mnar_10K_25, df_mnar_100K_25, df_mnar_1K_25_50, df_mnar_10K_25_50, df_mnar_100K_25_50,
             df_mnar_1K_25_75, df_mnar_10K_25_75, df_mnar_100K_25_75]

column_indices = [5]
missing_percentages = [50]
column_index = 7

for dataset in datasets_50:
    nan_count = dataset.iloc[:, column_index].isnull().sum()
    
    if nan_count < int((missing_percentages[0] / 100) * dataset.shape[0]):

        if dataset.shape[0] == 1000:
            # Create a copy of mnar_1K_25_df
            mnar_1K_25_df_copy = df_mnar_1K_25.copy()
            mnar_1K_50_25 = introduce_mnar_50(dataset, column_indices, missing_percentages)
            df_mnar_1K_50_25 = pd.DataFrame(mnar_1K_50_25, columns=dataset.columns)

            mnar_1K_nan_5_50_25 = df_mnar_1K_50_25[df_mnar_1K_50_25.columns[5]].isnull().sum()
            mnar_1K_nan_7_50_25 = df_mnar_1K_50_25[df_mnar_1K_50_25.columns[7]].isnull().sum()

            print("Number of NaN values 5 in dataset:", mnar_1K_nan_5_50_25)
            print("Number of NaN values 7 in dataset:", mnar_1K_nan_7_50_25)
        
        elif dataset.shape[0] == 10000:
            # Create a copy of mnar_10K_25_df
            mnar_10K_25_df_copy = df_mnar_10K_25.copy()
            mnar_10K_50_25 = introduce_mnar_50(dataset, column_indices, missing_percentages)
            df_mnar_10K_50_25 = pd.DataFrame(mnar_10K_50_25, columns=dataset.columns)

            mnar_10K_nan_5_50_25 = df_mnar_10K_50_25[df_mnar_10K_50_25.columns[5]].isnull().sum()
            mnar_10K_nan_7_50_25 = df_mnar_10K_50_25[df_mnar_10K_50_25.columns[7]].isnull().sum()

            print("Number of NaN values 5 in dataset:", mnar_10K_nan_5_50_25)
            print("Number of NaN values 7 in dataset:", mnar_10K_nan_7_50_25)
            
        elif dataset.shape[0] == 100000:
             # Create a copy of mnar_100K_25_df
             mnar_100K_25_df_copy = df_mnar_100K_25.copy()
             mnar_100K_50_25 = introduce_mnar_50(dataset, column_indices, missing_percentages)
             df_mnar_100K_50_25 = pd.DataFrame(mnar_100K_50_25, columns=dataset.columns)

             mnar_100K_nan_5_50_25 = df_mnar_100K_50_25[df_mnar_100K_50_25.columns[5]].isnull().sum()
             mnar_100K_nan_7_50_25 = df_mnar_100K_50_25[df_mnar_100K_50_25.columns[7]].isnull().sum()

             print("Number of NaN values 5 in dataset:", mnar_100K_nan_5_50_25)
             print("Number of NaN values 7 in dataset:", mnar_100K_nan_7_50_25)
             
    if nan_count == int((missing_percentages[0] / 100) * dataset.shape[0]):
       
        if dataset.shape[0] == 1000:
             # Create a copy of mnar_1K_25_50_df
             mnar_1K_25_50_df_copy = df_mnar_1K_25_50.copy()
             mnar_1K_50_50 = introduce_mnar_50(dataset, column_indices, missing_percentages)
             df_mnar_1K_50_50 = pd.DataFrame(mnar_1K_50_50, columns=dataset.columns)

             mnar_1K_nan_5_50_50 = df_mnar_1K_50_50[df_mnar_1K_50_50.columns[5]].isnull().sum()
             mnar_1K_nan_7_50_50 = df_mnar_1K_50_50[df_mnar_1K_50_50.columns[7]].isnull().sum()

             print("Number of NaN values 5 in dataset:", mnar_1K_nan_5_50_50)
             print("Number of NaN values 7 in dataset:", mnar_1K_nan_7_50_50)
       
        elif dataset.shape[0] == 10000: 
             # Create a copy of mnar_10K_25_50_df
             mnar_10K_25_50_df_copy = df_mnar_10K_25_50.copy()
             mnar_10K_50_50 = introduce_mnar_50(dataset, column_indices, missing_percentages)
             df_mnar_10K_50_50 = pd.DataFrame(mnar_10K_50_50, columns=dataset.columns)

             mnar_10K_nan_5_50_50 = df_mnar_10K_50_50[df_mnar_10K_50_50.columns[5]].isnull().sum()
             mnar_10K_nan_7_50_50 = df_mnar_10K_50_50[df_mnar_10K_50_50.columns[7]].isnull().sum()

             print("Number of NaN values 5 in dataset:", mnar_10K_nan_5_50_50)
             print("Number of NaN values 7 in dataset:", mnar_10K_nan_7_50_50)
             
        elif dataset.shape[0] == 100000:
             # Create a copy of mnar_100K_25_50_df
             mnar_100K_25_50_df_copy = df_mnar_100K_25_50.copy()
             mnar_100K_50_50 = introduce_mnar_50(dataset, column_indices, missing_percentages)
             df_mnar_100K_50_50 = pd.DataFrame(mnar_100K_50_50, columns=dataset.columns)

             mnar_100K_nan_5_50_50 = df_mnar_100K_50_50[df_mnar_100K_50_50.columns[5]].isnull().sum()
             mnar_100K_nan_7_50_50 = df_mnar_100K_50_50[df_mnar_100K_50_50.columns[7]].isnull().sum()

             print("Number of NaN values 5 in dataset:", mnar_100K_nan_5_50_50)
             print("Number of NaN values 7 in dataset:", mnar_100K_nan_7_50_50)
             
    if nan_count > int((missing_percentages[0] / 100) * dataset.shape[0]):
        
        if dataset.shape[0] == 1000:
             # Create a copy of mnar_1K_25_75_df
             mnar_1K_25_75_df_copy = df_mnar_1K_25_75.copy()
             mnar_1K_50_75 = introduce_mnar_50(dataset, column_indices, missing_percentages)
             df_mnar_1K_50_75 = pd.DataFrame(mnar_1K_50_75, columns=dataset.columns)

             mnar_1K_nan_5_50_75 = df_mnar_1K_50_75[df_mnar_1K_50_75.columns[5]].isnull().sum()
             mnar_1K_nan_7_50_75 = df_mnar_1K_50_75[df_mnar_1K_50_75.columns[7]].isnull().sum()

             print("Number of NaN values 5 in dataset:", mnar_1K_nan_5_50_75)
             print("Number of NaN values 7 in dataset:", mnar_1K_nan_7_50_75)
             
        elif dataset.shape[0] == 10000:
             # Create a copy of mnar_10K_25_75_df
             mnar_10K_25_75_df_copy = df_mnar_10K_25_75.copy()
             mnar_10K_50_75 = introduce_mnar_50(dataset, column_indices, missing_percentages)
             df_mnar_10K_50_75 = pd.DataFrame(mnar_10K_50_75, columns=dataset.columns)

             mnar_10K_nan_5_50_75 = df_mnar_10K_50_75[df_mnar_10K_50_75.columns[5]].isnull().sum()
             mnar_10K_nan_7_50_75 = df_mnar_10K_50_75[df_mnar_10K_50_75.columns[7]].isnull().sum()

             print("Number of NaN values 5 in dataset:", mnar_10K_nan_5_50_75)
             print("Number of NaN values 7 in dataset:", mnar_10K_nan_7_50_75)
             
        elif dataset.shape[0] == 100000:
             # Create a copy of mnar_100K_25_75_df
             mnar_100K_25_75_df_copy = df_mnar_100K_25_75.copy()
             mnar_100K_50_75 = introduce_mnar_50(dataset, column_indices, missing_percentages)
             df_mnar_100K_50_75 = pd.DataFrame(mnar_100K_50_75, columns=dataset.columns)

             mnar_100K_nan_5_50_75 = df_mnar_100K_50_75[df_mnar_100K_50_75.columns[5]].isnull().sum()
             mnar_100K_nan_7_50_75 = df_mnar_100K_50_75[df_mnar_100K_50_75.columns[7]].isnull().sum()

             print("Number of NaN values 5 in dataset:", mnar_100K_nan_5_50_75)
             print("Number of NaN values 7 in dataset:", mnar_100K_nan_7_50_75)
        

# For loop for each dataset to introduce 75% MNAR for TV1
datasets_75 = [df_mnar_1K_50_25, df_mnar_10K_50_25, df_mnar_100K_50_25, df_mnar_1K_50_50, df_mnar_10K_50_50, df_mnar_100K_50_50,
             df_mnar_1K_50_75, df_mnar_10K_50_75, df_mnar_100K_50_75]

column_indices = [5]
missing_percentages = [75]
column_index = 7

for dataset in datasets_75:
    nan_count = dataset.iloc[:, column_index].isnull().sum()
    
    if nan_count == int((missing_percentages[0] / 100) * dataset.shape[0]) / 3:
        
        if dataset.shape[0] == 1000:
            # Create a copy of mnar_1K_50_25_df
            mnar_1K_50_25_df_copy = df_mnar_1K_50_25.copy()
            mnar_1K_75_25 = introduce_mnar_75(dataset, column_indices, missing_percentages)
            df_mnar_1K_75_25 = pd.DataFrame(mnar_1K_75_25, columns=dataset.columns)

            mnar_1K_nan_5_75_25 = df_mnar_1K_75_25[df_mnar_1K_75_25.columns[5]].isnull().sum()
            mnar_1K_nan_7_75_25 = df_mnar_1K_75_25[df_mnar_1K_75_25.columns[7]].isnull().sum()

            print("Number of NaN values 5 in dataset:", mnar_1K_nan_5_75_25)
            print("Number of NaN values 7 in dataset:", mnar_1K_nan_7_75_25)
        
        elif dataset.shape[0] == 10000:
            # Create a copy of mnar_10K_50_25_df
            mnar_10K_50_25_df_copy = df_mnar_10K_50_25.copy()
            mnar_10K_75_25 = introduce_mnar_75(dataset, column_indices, missing_percentages)
            df_mnar_10K_75_25 = pd.DataFrame(mnar_10K_75_25, columns=dataset.columns)

            mnar_10K_nan_5_75_25 = df_mnar_10K_75_25[df_mnar_10K_75_25.columns[5]].isnull().sum()
            mnar_10K_nan_7_75_25 = df_mnar_10K_75_25[df_mnar_10K_75_25.columns[7]].isnull().sum()

            print("Number of NaN values 5 in dataset:", mnar_10K_nan_5_75_25)
            print("Number of NaN values 7 in dataset:", mnar_10K_nan_7_75_25)
            
        elif dataset.shape[0] == 100000:
             # Create a copy of mnar_100K_50_25_df
             mnar_100K_50_25_df_copy = df_mnar_100K_50_25.copy()
             mnar_100K_75_25 = introduce_mnar_75(dataset, column_indices, missing_percentages)
             df_mnar_100K_75_25 = pd.DataFrame(mnar_100K_75_25, columns=dataset.columns)

             mnar_100K_nan_5_75_25 = df_mnar_100K_75_25[df_mnar_100K_75_25.columns[5]].isnull().sum()
             mnar_100K_nan_7_75_25 = df_mnar_100K_75_25[df_mnar_100K_75_25.columns[7]].isnull().sum()

             print("Number of NaN values 5 in dataset:", mnar_100K_nan_5_75_25)
             print("Number of NaN values 7 in dataset:", mnar_100K_nan_7_75_25)
        
    if nan_count == int((missing_percentages[0] / 100) * dataset.shape[0]) / 1.5:
        
        if dataset.shape[0] == 1000:
            # Create a copy of mnar_1K_50_50_df
            mnar_1K_50_50_df_copy = df_mnar_1K_50_50.copy()
            mnar_1K_75_50 = introduce_mnar_75(dataset, column_indices, missing_percentages)
            df_mnar_1K_75_50 = pd.DataFrame(mnar_1K_75_50, columns=dataset.columns)

            mnar_1K_nan_5_75_50 = df_mnar_1K_75_50[df_mnar_1K_75_50.columns[5]].isnull().sum()
            mnar_1K_nan_7_75_50 = df_mnar_1K_75_50[df_mnar_1K_75_50.columns[7]].isnull().sum()

            print("Number of NaN values 5 in dataset:", mnar_1K_nan_5_75_50)
            print("Number of NaN values 7 in dataset:", mnar_1K_nan_7_75_50)
            
        elif dataset.shape[0] == 10000:
            # Create a copy of mnar_10K_50_50_df
            mnar_10K_50_50_df_copy = df_mnar_10K_50_50.copy()
            mnar_10K_75_50 = introduce_mnar_75(dataset, column_indices, missing_percentages)
            df_mnar_10K_75_50 = pd.DataFrame(mnar_10K_75_50, columns=dataset.columns)

            mnar_10K_nan_5_75_50 = df_mnar_10K_75_50[df_mnar_10K_75_50.columns[5]].isnull().sum()
            mnar_10K_nan_7_75_50 = df_mnar_10K_75_50[df_mnar_10K_75_50.columns[7]].isnull().sum()

            print("Number of NaN values 5 in dataset:", mnar_10K_nan_5_75_50)
            print("Number of NaN values 7 in dataset:", mnar_10K_nan_7_75_50)
            
        elif dataset.shape[0] == 100000:
            # Create a copy of mnar_100K_50_50_df
            mnar_100K_50_50_df_copy = df_mnar_100K_50_50.copy()
            mnar_100K_75_50 = introduce_mnar_75(dataset, column_indices, missing_percentages)
            df_mnar_100K_75_50 = pd.DataFrame(mnar_100K_75_50, columns=dataset.columns)

            mnar_100K_nan_5_75_50 = df_mnar_100K_75_50[df_mnar_100K_75_50.columns[5]].isnull().sum()
            mnar_100K_nan_7_75_50 = df_mnar_100K_75_50[df_mnar_100K_75_50.columns[7]].isnull().sum()

            print("Number of NaN values 5 in dataset:", mnar_100K_nan_5_75_50)
            print("Number of NaN values 7 in dataset:", mnar_100K_nan_7_75_50)
            
    if nan_count == int((missing_percentages[0] / 100) * dataset.shape[0]):
        
        if dataset.shape[0] == 1000:
            # Create a copy of mnar_1K_50_75_df
            mnar_1K_50_75_df_copy = df_mnar_1K_50_75.copy()
            mnar_1K_75_75 = introduce_mnar_75(dataset, column_indices, missing_percentages)
            df_mnar_1K_75_75 = pd.DataFrame(mnar_1K_75_75, columns=dataset.columns)

            mnar_1K_nan_5_75_75 = df_mnar_1K_75_75[df_mnar_1K_75_75.columns[5]].isnull().sum()
            mnar_1K_nan_7_75_75 = df_mnar_1K_75_75[df_mnar_1K_75_75.columns[7]].isnull().sum()

            print("Number of NaN values 5 in dataset:", mnar_1K_nan_5_75_75)
            print("Number of NaN values 7 in dataset:", mnar_1K_nan_7_75_75)
            
        if dataset.shape[0] == 10000:
            # Create a copy of mnar_10K_50_75_df
            mnar_10K_50_75_df_copy = df_mnar_10K_50_75.copy()
            mnar_10K_75_75 = introduce_mnar_75(dataset, column_indices, missing_percentages)
            df_mnar_10K_75_75 = pd.DataFrame(mnar_10K_75_75, columns=dataset.columns)

            mnar_10K_nan_5_75_75 = df_mnar_10K_75_75[df_mnar_10K_75_75.columns[5]].isnull().sum()
            mnar_10K_nan_7_75_75 = df_mnar_10K_75_75[df_mnar_10K_75_75.columns[7]].isnull().sum()

            print("Number of NaN values 5 in dataset:", mnar_10K_nan_5_75_75)
            print("Number of NaN values 7 in dataset:", mnar_10K_nan_7_75_75)
            
        if dataset.shape[0] == 100000:
            # Create a copy of mnar_10K_50_75_df
            mnar_100K_50_75_df_copy = df_mnar_100K_50_75.copy()
            mnar_100K_75_75 = introduce_mnar_75(dataset, column_indices, missing_percentages)
            df_mnar_100K_75_75 = pd.DataFrame(mnar_100K_75_75, columns=dataset.columns)

            mnar_100K_nan_5_75_75 = df_mnar_100K_75_75[df_mnar_100K_75_75.columns[5]].isnull().sum()
            mnar_100K_nan_7_75_75 = df_mnar_100K_75_75[df_mnar_100K_75_75.columns[7]].isnull().sum()

            print("Number of NaN values 5 in dataset:", mnar_100K_nan_5_75_75)
            print("Number of NaN values 7 in dataset:", mnar_100K_nan_7_75_75)
        
