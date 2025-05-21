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

    2025/05/20
  
    - I plotted learning curves for three types of models - linear Logistic Regression, RandomForest, gradient boosting (lightGBM).
        - lightGBM and logistic regression are underfitting, and their validation errors are plateauing very early (with 10000 data points, instead of 69000 that are in the training set.)
        - random forest is overfitting the data a lot, so not using it going forward.
    - I looked at highly correlated features and removed them. I also tried recursive feature elimination and that showed that a maximum of 10 features is good enough to reach max performance in logistic regression.
    - I also tried to use selectKBest features and that also does not improve the model much  with increasing number of features.
    -   - Selected a small number of features - 3 - to examine the errors in the model in detail. Even with three features, the f1 score for logistic regression and lightGBM is 0.36.
    - I performed error analysis on this model and found that for lightGBM, the performance of the model is maximized. because the features are all categorical, the model has to choose combinations of features to make a prediction. As long as the model is choosing the combination that maximizes precision and recall, it works and lightGBM seems to be doing that.
    - I also tried balancing the dataset with RandomUnderSampling and that also does not make any difference in the performance.
    - Because all features are categorical, there is a limit to max performance, which is clear using a small number of variables.
    - I don't think combining models would do much in this case.
    - I think I will also check a few other models like kNN classifier and SVC, and perhaps try a dimensional reduction method, do error analysis on those with the 3 feature dataset and finalize the model. - TO DO

    
   
