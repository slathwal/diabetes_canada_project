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
    - In progress
