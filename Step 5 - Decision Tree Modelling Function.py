#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:49:23 2023

@author: leemingjun
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def train_prediction_model(dataset, target_col, dataset_name):
    
    np.random.seed(42)
    
    if "DELETE" in dataset_name:
        
        # Create a new DataFrame without the target column
        X = dataset.drop(target_col, axis=1)
        
        # Set Dup_Index as index
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
        
        
        ########### One-hot encoding for the target variable ###########

        target_encoder = OneHotEncoder(sparse=False)
        y_encoded = pd.DataFrame(target_encoder.fit_transform(y.values.reshape(-1, 1)), index=y.index)
        y_encoded.columns = target_encoder.get_feature_names_out([target_col])


        ########### Train the Decision Tree model ###########

        # Split the data into training and testing sets at 70:30
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)

        # Create an instance of the DecisionTreeClassifier
        tree = DecisionTreeClassifier()

        # Fit the decision tree model
        tree.fit(X_train, y_train)
        
        
        # Inverse transform y_test
        y_test_categorical_inverse = pd.DataFrame(target_encoder.inverse_transform(y_test), index=y_test.index)
        y_test_categorical_inverse.columns = [target_col]

        # Sort the index in ascending order
        y_test_categorical_inverse = y_test_categorical_inverse.sort_index()
        
        # Save the trained model and encoder for later prediction
        model_info = {
            'tree': tree,
            'encoder': encoder,
            'target_encoder': target_encoder,
            'categorical_cols': categorical_cols,
            'target_col': target_col,
            'X_test': X_test,
            'encoded_column_names': encoded_column_names
           }
        
        return model_info, y_test_categorical_inverse
    
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
        
        
        ########### One-hot encoding for the target variable ###########

        target_encoder = OneHotEncoder(sparse=False)
        y_encoded = pd.DataFrame(target_encoder.fit_transform(y.values.reshape(-1, 1)))
        y_encoded.columns = target_encoder.get_feature_names_out([target_col])


        ########### Train the Decision Tree model ###########

        # Split the data into training and testing sets at 70:30
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)

        # Create an instance of the DecisionTreeClassifier
        tree = DecisionTreeClassifier()

        # Fit the decision tree model
        tree.fit(X_train, y_train)
        
        # Inverse transform X_test
        y_test_categorical_inverse = pd.DataFrame(target_encoder.inverse_transform(y_test), index=y_test.index)
        y_test_categorical_inverse.columns = [target_col]

        # Sort the index in ascending order
        y_test_categorical_inverse = y_test_categorical_inverse.sort_index()
        
        # Save the trained model and encoder for later prediction
        model_info = {
            'tree': tree,
            'encoder': encoder,
            'target_encoder': target_encoder,
            'categorical_cols': categorical_cols,
            'target_col': target_col,
            'X_test': X_test,
            'encoded_column_names': encoded_column_names
           }
        
        return model_info, y_test_categorical_inverse



def make_predictions(dataset, model_info, target_col):
    
    np.random.seed(42)
    
    # Retrieve the model information
    tree = model_info['tree']
    encoder = model_info['encoder']
    target_encoder = model_info['target_encoder']
    categorical_cols = model_info['categorical_cols']
    target_col = model_info['target_col']
    X_test = model_info['X_test']
    encoded_column_names = model_info['encoded_column_names']
    
    # Copy the original treated dataset
    data_predicted = X_test.copy()

    # Drop the columns in encoded_column_names
    data_predicted = data_predicted.drop(encoded_column_names, axis=1)

    # Sort the index in ascending order
    data_predicted = data_predicted.sort_index()
    
    # Make predictions using the trained model
    predictions = tree.predict(X_test)

    # Inverse transform X_test
    X_test_categorical_inverse = pd.DataFrame(encoder.inverse_transform(X_test[encoded_column_names]), index=X_test.index)
    X_test_categorical_inverse.columns = categorical_cols

    # Sort the index in ascending order
    X_test_categorical_inverse = X_test_categorical_inverse.sort_index()

    # Replace the predicted column(s) in the data_predicted with the predictions
    # Merge the inverse transformed categorical columns back into data_predicted
    data_predicted[categorical_cols] = X_test_categorical_inverse
    data_predicted[target_col] = target_encoder.inverse_transform(predictions)

    # Rearrange the columns in the desired order
    desired_order = ["N_Var1", "N_Var2", "N_Var3", "N_Var4", "C_Var1", "C_Var2", "C_Var3", "C_Var4"]
    data_predicted = data_predicted[desired_order]

    return data_predicted


