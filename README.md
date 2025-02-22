# Titanic Survival Prediction with Fairness Analysis

## Overview

This project focuses on building a machine learning model to predict survival outcomes on the Titanic, using features such as age, gender, passenger class, and fare. A strong emphasis is placed on fairness analysis to ensure the model performs equitably across different subgroups, such as gender and passenger class.

The dataset used in this project is the Titanic dataset, which contains information about 891 passengers, including whether they survived or not. The project involves data preprocessing, feature engineering, model training, and fairness evaluation.

### Dataset

The dataset used in this project is the Titanic dataset, which can be downloaded from Kaggle. It contains the following key features:

- **Survived**: Binary target variable (1 = survived, 0 = did not survive).
- **Pclass**: Passenger class (1 = 1st class, 2 = 2nd class, 3 = 3rd class).
- **Sex**: Gender of the passenger (male/female).
- **Age**: Age of the passenger.
- **SibSp**: Number of siblings/spouses aboard.
- **Parch**: Number of parents/children aboard.
- **Fare**: Fare paid for the ticket.
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

### Project Steps

1. **Data Preprocessing**:
- Handle missing values (e.g., impute age with the median).
- Encode categorical variables (e.g., one-hot encoding for Embarked and Title).
- Create new features (e.g., FamilySize, Fare_Bin, Children).

2. **Model Training**:
- Split the data into training and testing sets.
- Train Logistic Regression and Random Forest models.
- Evaluate model performance using accuracy, precision, recall, and F1-score.

3. **Fairness Analysis**:
   
Evaluate fairness across gender and passenger class using metrics such as:
- Equalized Odds Difference
- Demographic Parity Difference
- Selection Rate
- Disparate Impact
- Visualize fairness disparities using bar plots.

4. **Visualizations**:
- Plot confusion matrices for both models.
- Generate ROC curves and precision-recall curves.
- Use SHAP values to explain model predictions.

### Results

**Model Performance**:
- Logistic Regression achieved an accuracy of 84.36%.
- Random Forest achieved an accuracy of 80.45%.

**Fairness Insights**:
- The model is biased toward females, with a high false positive rate (62.5%) and selection rate (90.16%).
- The model is fair across passenger classes, with no significant disparities in performance.

**Disparate Impact**:
- **Gender**: 0.132 (significant bias).
- **Passenger Class**: 1.0 (no bias).

### Future Work

**Mitigate Gender Bias**:
- Use fairness-aware algorithms (e.g., fairlearn's GridSearch with fairness constraints).
- Adjust class weights or resample the dataset to reduce disparities.

**Extend Fairness Analysis**:
- Evaluate fairness across other sensitive attributes (e.g., age groups, family size).

**Improve Model Performance**:
- Experiment with advanced models (e.g., XGBoost, LightGBM).
- Perform hyperparameter tuning to optimize model performance.

### Source

https://www.kaggle.com/datasets/yasserh/titanic-dataset
