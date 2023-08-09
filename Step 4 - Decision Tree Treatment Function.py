#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:56:00 2023

@author: leemingjun
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def impute_missing_values(dataset, missing_cols):
    
    np.random.seed(42)
    
    # Create a new DataFrame to store the predicted values
    data_predicted = dataset.copy()

    # Flag to track if there are still missing values
    missing_values_exist = True

    while missing_values_exist:
        # Iterate over each missing column
        for col in missing_cols:
            # Separate the dataset into two parts: one with missing values and one without
            data_missing = data_predicted[data_predicted[col].isnull()]
            data_complete = data_predicted.dropna(subset=[col])

            # Create a new DataFrame to store the predicted values for the current testing variable
            data_predicted_col = data_missing.copy()

            # Create a new DataFrame without the current missing column
            X = data_complete.drop(missing_cols, axis=1)

            # Extract the target variable (missing column)
            y = data_complete[col]

            ########### One-hot encoding for NON-MISSING rows on NON-MISSING columns ###########

            # Identify categorical variables and perform one-hot encoding
            categorical_cols = X.select_dtypes(include='object').columns
            encoder = OneHotEncoder(sparse=False)
            X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))

            X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
            X_encoded = pd.concat([X.drop(categorical_cols, axis=1).reset_index(drop=True), X_encoded], axis=1)

            ########### Train the Decision Tree model ###########

            # Split the data into training and testing sets at 70:30
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

            # Create an instance of the DecisionTreeClassifier
            tree = DecisionTreeClassifier()

            # Fit the decision tree model
            tree.fit(X_train, y_train)

            ########### One-hot encoding for MISSING rows on NON-MISSING columns ###########

            # Perform the same encoding for the missing values for the current column
            missing_values = data_missing.drop(missing_cols, axis=1)
            missing_values_encoded = pd.DataFrame(encoder.transform(missing_values[categorical_cols]))
            missing_values_encoded.columns = encoder.get_feature_names_out(categorical_cols)
            missing_values_encoded = pd.concat([missing_values.drop(categorical_cols, axis=1).reset_index(drop=True),
                                                missing_values_encoded], axis=1)

            # Predict the missing values for the current column
            predicted_values = tree.predict(missing_values_encoded)

            # Replace the missing values for the current column with the predicted values
            data_predicted_col[col] = predicted_values

            # Update the predicted values in the main DataFrame
            data_predicted.update(data_predicted_col)

        # Check if there are any remaining missing values after imputation
        missing_values_count = data_predicted.isnull().sum().sum()
        if missing_values_count == 0:
            missing_values_exist = False

    return data_predicted

