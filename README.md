# Diabetes Prediction using Canadian Community Health Survey Data

This project explores and builds a predictive model for identifying individuals at risk for diabetes, using data from the [**Canadian Community Health Survey (CCHS) Public Use Microdata File (PUMF)**](https://www150.statcan.gc.ca/n1/en/catalogue/82M0013X) for the years 2019-2020. The project follows standard machine learning best practices, including data exploration, preprocessing, model training, evaluation, and cross-validation.

## Methodology

1. **Exploratory Data Analysis (EDA)**
   - Examined data distributions, correlations, and potential predictors
   - Looked at prevalence of diabetes across provinces, by age group, by sex, by household income.
   - Identified important variables
   - Handled missing values
   - Developed a baseline model
        - One-hot encoding of categorical variables
        - Standardization or normalization where applicable
        - Boolean outcome variable for diabetes diagnosis
        - Evaluation metrics: F1 score, precision, recall
        - Handling class imbalance
        - Tried Logistic Regression, SVC and RandomForest

2. **Improving the model(s)**
    - So far (2025/05/17) the performance of linear models (Logistic Regression) is similar to more complex models (RandomForest and XGBoost) 
    - Fine-tuning the models:
        - Tried XGBoost, RandomForest, LogisticRegression - the performance of RandomForest improved with finetuning, reaching an f1 score of 0.42
    - Better data pre-processing
        - Treat ordinal features as categorical - no change
        - Remove features that have low variance - no features with low variance
        - Remove features that are highly correlated - TO DO
            - Looked at what measures of correlation to use for ordinal and categorical/nominal data.
        - Peform automated feature selection - TO DO
    - Error Analysis - TO DO
    - Look at learning curves on training and validation data - TO DO
    - Try combining models to improve performance - TO DO
   
