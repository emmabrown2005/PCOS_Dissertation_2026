# PCOS_Dissertation_2026

## Overview
An end-to-end machine learning pipeline for predicting Polycystic Ovary Syndrome (PCOS)
from clinical and biometric features. The project compares a wide range of classifiers
across multiple preprocessing strategies, selects an optimal configuration, and validates
it with explainability and learning-curve diagnostics.
The work is implemented in a single Jupyter notebook (PCOSProject_Enhanced.ipynb) and is
designed to be reproducible, leakage-free, and focused on recall (PCOS sensitivity) —
prioritising the correct identification of positive cases over raw accuracy.

## Data acquisition & preprocessing

Loads the extended PCOS dataset.
Drops non-predictive identifier columns (e.g. serial / file numbers) to avoid spurious
correlations.
Sanitises feature names so they are safe for gradient-boosting libraries (LightGBM).
Defines PCOS_YN as the binary target.


## Exploratory data analysis

Class-distribution visualisation and class-balance ratio.
Per-feature correlation with the target and a full correlation heatmap.


## Train / test split

80 / 20 stratified split preserving the class ratio; the test set is reserved for final
hold-out evaluation only.


## Feature selection

Mutual Information (top-k features).
Boruta (Random-Forest wrapper against shadow features).
Union of both methods, followed by removal of highly collinear features
(correlation > 0.90).
Produces four comparable feature sets: All, MI, Boruta, Union.


## Resampling strategy

Comparison of SMOTE variants (Standard, Borderline, KMeans) under cross-validation.
SMOTE is always the first pipeline step so it is applied inside each fold only,
preventing leakage into validation data.


## Model training & cross-validation

Twelve classifiers benchmarked: Logistic Regression, KNN, SVM, a narrow neural network
(MLP), Decision Tree, Naive Bayes, Random Forest, AdaBoost, Gradient Boosting, XGBoost,
LightGBM, and CatBoost.
Scaling applied only to the models that require it (LR, SVM, KNN, MLP).
Evaluated with 5-fold stratified cross-validation on accuracy, recall, F1, and ROC-AUC.


## Experiment matrix (E1–E7)

Systematic ablation across the combinations of SMOTE, PCA, and
hyperparameter tuning, summarised in accuracy / recall heatmaps and a best-per-model
ranking.


## Hyperparameter tuning

GridSearchCV / RandomizedSearchCV per model, optimised for recall.
Best estimators retained for downstream ensembling and explainability.


## Ensemble methods

Stacking classifier (Random Forest, XGBoost, LightGBM, CatBoost → Logistic Regression
meta-learner).
Soft-voting ensemble of the top tuned models.


# Explainability (SHAP)

Beeswarm summary, mean-absolute-importance bar chart, and dependence plots to interpret
which clinical features drive PCOS predictions.


## Model diagnostics

Learning curves for the tuned Random Forest on both accuracy and recall, used to
confirm the final configuration generalises well and shows no meaningful overfitting
(training and cross-validation scores converge as data grows).
