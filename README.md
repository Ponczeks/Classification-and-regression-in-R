# Classification-and-regression-in-R
# Comparative Analysis of Machine Learning Models for Classification and Regression in R

This repository contains an R-based project comparing the performance of decision trees and random forests for classification and regression tasks. The analysis includes:

- **Classification**: Predicting heart disease using the `heart.csv` dataset with decision trees (`rpart`) and random forests (`randomForest`). Models are tuned via grid search and evaluated using cross-validation with metrics such as Accuracy, AUC, Precision, and Recall. Visualizations include confusion matrix heatmaps.
- **Regression**: Forecasting gold prices from the `financial_regression.csv` dataset using decision trees and random forests. Hyperparameters are optimized with grid search, and performance is assessed with RMSE, MAE, R², and MAPE. Results are visualized with scatter plots and regression trees.

## Contents
- **Script**: `script.R` - Full R code implementing data preprocessing, model training, tuning, evaluation, and visualization.
- **Data**: 
  - `heart.csv` - Dataset for classification (not included; source required).
  - `financial_regression.csv` - Dataset for regression (not included; source required).
- **Documentation**: Theoretical background and results summary (embedded in the script and original OCR document).

## Key Features
- Data preprocessing: Handling missing values and categorical variables.
- Hyperparameter tuning via grid search and cross-validation.
- Comprehensive evaluation with multiple performance metrics.
- Visualizations for model interpretation (e.g., decision trees, heatmaps, scatter plots).

## Requirements
- R packages: `tidyverse`, `caret`, `rpart`, `rpart.plot`, `randomForest`, `pROC`, `ggplot2`, `dplyr`.
- Datasets: `heart.csv` and `financial_regression.csv` (users must provide their own copies).

## Usage
1. Install required R packages.
2. Load the datasets (`heart.csv` and `financial_regression.csv`) into your working directory.
3. Run `script.R` to execute the analysis and generate results.

