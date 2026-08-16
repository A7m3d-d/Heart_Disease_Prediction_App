# Heart_Disease_Prediction_App
A machine learning web application for predicting cardiovascular disease using Logistic Regression and Gradient Boosting models, with an interactive Streamlit interface for real-time predictions.

## Project Overview

Cardiovascular disease is one of the major health challenges worldwide. This project uses patient medical examination data to develop machine learning models that can predict whether a patient is likely to have cardiovascular disease.

The project focuses on building and evaluating classification models while providing an interactive Streamlit application that allows users to enter patient information and obtain a prediction.

## Problem Statement

Cardiovascular disease can be influenced by multiple patient-related factors, including age, blood pressure, cholesterol, glucose level, body measurements, and lifestyle factors.

The challenge is to use these medical examination features to identify patterns associated with cardiovascular disease and build a classification system that can support early risk assessment.

## Project Objective

The main objective of this project is to develop and evaluate machine learning classification models for cardiovascular disease prediction and deploy the trained models through an interactive Streamlit web application.

The project aims to:
- Prepare and analyze patient medical data.
- Build and evaluate classification models.
- Compare Logistic Regression and Gradient Boosting performance.
- Provide an interactive interface for making predictions.

 ## Dataset

The project uses the Cardiovascular Disease dataset, which contains **70,000 patient records** and **13 columns**.

The dataset includes patient demographic information, physical measurements, medical examination results, and lifestyle-related features. The target variable, `cardio`, represents the presence or absence of cardiovascular disease:

- `0` — No cardiovascular disease
- `1` — Cardiovascular disease

The dataset is used for data preprocessing, exploratory data analysis, feature engineering, and machine learning model development.

## Data Preprocessing

The dataset was prepared through several preprocessing steps to improve data quality and ensure that the machine learning models receive suitable input features.

The main preprocessing steps included:

- Checking for missing values and duplicate records.
- Removing the non-predictive `id` column.
- Converting age from days into years.
- Creating a BMI feature using height and weight.
- Filtering biologically implausible values and extreme measurements.
- Separating numerical and categorical features.
- Applying `StandardScaler` to numerical features.
- Applying `OneHotEncoder` to categorical features.
- Using a `ColumnTransformer` to combine the preprocessing steps into a consistent pipeline.
- Splitting the data into training and testing sets while maintaining class proportions.

  ## Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the structure of the dataset, examine feature distributions, identify potential outliers, and explore relationships between patient characteristics and cardiovascular disease.

The analysis included:

- Examining the distribution of the target variable.
- Analyzing numerical and categorical features.
- Using boxplots and distribution plots to identify patterns and potential outliers.
- Examining relationships between medical features and cardiovascular disease.
- Analyzing correlations between numerical variables.

The EDA helped identify patterns in features such as age, blood pressure, cholesterol, glucose, height, and weight.

## Machine Learning Models

Two classification models were used for the final prediction system:

### Logistic Regression

Logistic Regression was used as a classification model for predicting the presence or absence of cardiovascular disease.

### Gradient Boosting

Gradient Boosting was used as a more advanced classification model capable of learning complex relationships between the input features and the target variable.

Both models were trained using the prepared dataset and evaluated using classification performance metrics.

## Model Evaluation

The models were evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

The evaluation results were used to compare the performance of the models and assess their effectiveness in predicting cardiovascular disease.

## Streamlit Application

The trained machine learning models were integrated into an interactive Streamlit web application.

The application allows users to enter patient information, including:

- Age
- Gender
- Height
- Weight
- Systolic blood pressure
- Diastolic blood pressure
- Cholesterol
- Glucose
- Smoking
- Alcohol intake
- Physical activity

The entered information is processed and passed to the trained machine learning models to generate a cardiovascular disease prediction.

## Project Structure

```text
Cardiovascular_Heart_Disease_Prediction/
│
├── app.py
├── cardio_train.csv
├── FInal_project_models_updated (1).ipynb
├── gradient_boosting_model.pkl
├── logistic_model.pkl
├── quantile_transformer.pkl
├── requirements.txt
├── README.md
└── LICENSE
