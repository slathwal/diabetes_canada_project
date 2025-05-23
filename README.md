Note: This file is being edited.

# Diabetes Prediction using Canadian Community Health Survey Data

This project explores and builds a predictive model for identifying individuals at risk for diabetes, using data from the [**Canadian Community Health Survey (CCHS) Public Use Microdata File (PUMF)**](https://www150.statcan.gc.ca/n1/en/catalogue/82M0013X) for the years 2019-2020. The survey is broad and contains more than 600 questions on a variety of topics such as geographical location, demographics, general health, general mental health, education, chronic conditions (including diabetes), lifestyle factors such as fruit and vegetable consumption, smoking, alcohol use, drug use, physical activity, sedentary behaviours, sexual behaviours etc., engagement with healthcare system, screening test, involvement with community, socio-demographic characteristics such as country of birth, immigration status etc., health insurance, food security, household income and many more.

- The project includes exploratory analysis to identify the best methods for data pre-processing, and implements machine learning methods such as Logistic Regression, SupportVectorClassifier and LightGBM to predict the presence of diabetes in survey respondents, and to identify the most important features asssociated with the prediction.

- Feature selection and undersampling are used to balance the classes in the dataset. 

- Precision recall curves and F1 score are used to assess and compare different models.

- The project shows that general purpose survey data can be used to recover important factors associated with chronic conditions such as diabetes, but the predictive models built on these data suffer from low precision when data are unbalanced (less positive cases compared to negative cases), i.e., a large number of false postives.

***
## Content
2. [Objective](#objectives)
3. [WIP - Methodology](#methodology)
4. [WIP - Results](#results)
5. [WIP - Working Notes](#working-notes)

## Objectives

The main goal of this project is to understand the following:

**Can a general community health survey data be used to extract useful information about factors associated with a chronic condition such as diabetes?**

## Methodology

**Exploratory Data Analysis**
1. Understand the variables present in the data
2. Filter the data to include only adults and people with known diabetes status.
3. Calculate missing values for each variable and remove variables with >30% missing data
4. Remove variables that have same value across all rows and that have different values across all rows, as well as variables that do not contain any useful information.
5. Identify ordinal and categorical variables, calculate correlation between all variables and drop highly correlated variables.
6. Undersample the data to balance classes.
7. Pre-process the data.
8. Fit a model
9. Evaluate the model fit and its performance.
10. Identify the most important features contributing to prediction of diabetes status.

## Results

1. Understanding the performance limit of a predicitve model on the data
2. Logistic regression model
3. Support Vector Classifier
4. LightGBM

## Working Notes

2025/05/21

- I calculate the maximum precision, recall and f1 score that can be obtained in the data manually. I was able to do this because my data consist of discrete value columns and it is possible to manually classify them. I picked two columns and calculated the theoretical maximum. 
- I then resampled the data and fit a logistic regression model with limited features
- To my surprise, I discovered that the f1 score was close to theoretical maximum if the classes were balanced.
- I am not sure how to calculate a theoretical maximum if classes are not balanced.
- I discovered that when I tried undersampling yesterday, I used it in a pipeline from the package `imblearn` and that was not working. If I resampled data separately and then built a logistic regression model on resampled, balanced data, I was able to achieve the theoretical maximum.
- I have added a notebook with the finalized steps.
- I have also put the data cleaning functions in the src folder and imported those functions in the notebook to use.

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
- I think I will also check a few other models like kNN classifier (kNN would not be good here because) and SVC, and perhaps try a dimensional reduction method, do error analysis on those with the 3 feature dataset and finalize the model. - TO DO

2025/05/17
- So far the performance of linear models (Logistic Regression) is similar to more complex models (RandomForest and XGBoost) 
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

2025/05/11-16

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

   


  

    
   
