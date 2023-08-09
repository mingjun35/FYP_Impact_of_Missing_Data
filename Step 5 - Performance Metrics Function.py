#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:37:28 2023

@author: leemingjun
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


def calculate_misclassification_rate(true_values, data_predicted, data_predicted_name):
    
    true_values, data_predicted = true_values.align(data_predicted, axis=0, copy=False)
    
    misclassified_indices = true_values[true_values != data_predicted[target_col]].index

    # Get the true values and predicted values for misclassified instances
    misclassified_true_values = true_values.loc[misclassified_indices]
    misclassified_predicted_values = data_predicted[target_col].loc[misclassified_indices]

    # Calculate the misclassification rate
    misclassified = len(misclassified_indices)
    total_instances = len(true_values)

    # Misclassification rate is division between num of misclassified and total instances
    misclassification_rate = misclassified / total_instances

    return misclassification_rate



def calculate_accuracy(true_values, data_predicted, data_predicted_name):
    
    true_values, data_predicted = true_values.align(data_predicted, axis=0, copy=False)
    
    accurate_indices = true_values[true_values == data_predicted[target_col]].index

    # Get the true values and predicted values for misclassified instances
    accurate_true_values = true_values.loc[accurate_indices]
    accurate_predicted_values = data_predicted[target_col].loc[accurate_indices]

    # Calculate the misclassification rate
    correct_predictions = len(accurate_indices)
    total_instances = len(true_values)

    # Misclassification rate is division between num of misclassified and total instances
    accuracy = correct_predictions / total_instances

    return accuracy



def get_confusion_matrix(true_values, data_predicted, data_predicted_name):
    
    true_values, data_predicted = true_values.align(data_predicted, axis=0, copy=False)
    
    # Extract the predicted values for the target column
    predicted_values = data_predicted[target_col]
    predicted_values = predicted_values.fillna("missing")

    # Get the unique classes from true values
    classes = true_values.unique()

    # Calculate the confusion matrix
    cm = confusion_matrix(true_values, predicted_values, labels=classes)
            
    return cm
                


def calculate_metrics(confusion_matrix_df):
        
    # Calculate precision
    precision = true_positive / (true_positive + false_positive)
        
    # Calculate sensitivity (recall)
    sensitivity = true_positive / (true_positive + false_negative)
        
    # Calculate specificity
    specificity = true_negative / (true_negative + false_positive)
        
    # Calculate F1-score
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
        
    return precision, sensitivity, specificity, f1_score


