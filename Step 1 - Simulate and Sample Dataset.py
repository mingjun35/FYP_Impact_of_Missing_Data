#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:35:43 2023

@author: leemingjun
"""
import numpy as np
import pandas as pd
import random

np.random.seed(10)

# Generating Dataset

# Numerical Variables
var1 = np.random.random(100000)
var2 = np.random.randint(1, 10000, 100000)
var3 = np.random.uniform(-1, 1, 100000)
var4 = np.random.randint(-10000, 1, 100000)

#C ategorical Variables
var5 = np.random. choice (a= ['Male', 'Female'], size=100000)
var6 = np.random.choice(a=['Blue', 'Green', 'Red'], size=100000)
var7 = np.random. choice (a=['Raining', 'Not_Raining'], size=100000)
var8 = np.random.choice(a=[ 'Mouse', 'Cat', 'Dog', 'Rabbit', 'Fish'], size=100000)

# Create a DataFrame to store the variables
complete_data = pd.DataFrame ({
    'N_Var1': var1,
    'N_Var2': var2,
    'N_Var3': var3,
    'N_Var4': var4,
    'C_Var1': var5,
    'C_Var2': var6,
    'C_Var3': var7,
    'C_Var4': var8,
})


# Descriptive Analysis of the complete dataset

# Select the categorical columns
complete_categorical_cols = complete_data.select_dtypes(include=['object'])  # Adjust the data types as per your dataset

# Calculate the frequency count and proportion for each categorical column
for col in complete_categorical_cols:
    print(f"\nDescriptive analysis for column: {col}")
    
    # Calculate the Frequency count of each unique value in the column
    frequency_count = complete_data[col].value_counts()
    print(frequency_count)
    
    # Calculate the Proportion of each unique values (Unique Count//Total Count)
    proportion = complete_data[col].value_counts(normalize=True)
    print(proportion)

#complete_data.to_csv('/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Complete_Data.csv')

#complete_data=pd.read_csv('/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Complete_Data.csv')


# Systematic Sampling of 1K & 10K data

# 1K Data

step_1K = 2
sample_size_1K = 1000

def systematic_sampling(complete_data, step, sample_size):
    total_rows = step * sample_size
    indexes = np.arange(0,total_rows,step=step)
    systematic_sample = complete_data.iloc[indexes[:sample_size]]
    
    return systematic_sample

sys_sample_1K = systematic_sampling(complete_data, step_1K, sample_size_1K)

print(sys_sample_1K)

#sys_sample_1K.to_csv('/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Sampled_1K.csv')

# 10K Data

step_10K = 8
sample_size_10K = 10000

def systematic_sampling(complete_data, step, sample_size):
    total_rows = step * sample_size
    indexes = np.arange(0,total_rows,step=step)
    systematic_sample = complete_data.iloc[indexes[:sample_size]]
    
    return systematic_sample

sys_sample_10K = systematic_sampling(complete_data, step_10K, sample_size_10K)

print(sys_sample_10K)

#sys_sample_10K.to_csv('/Users/leemingjun/Documents/Semester 9 - April 2023/Capstone Project 2/Sampled_10K.csv')
