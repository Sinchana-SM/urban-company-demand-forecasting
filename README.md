# urban-company-demand-forecasting
# Urban Company – Demand Forecasting using LSTM

## Project Overview

This project focuses on developing an end-to-end AI/ML demand forecasting pipeline for a home services marketplace inspired by Urban Company.

The objective is to predict future service booking demand for different cities and service categories using historical booking data, temporal patterns, weather information, and machine learning techniques.

The project uses a PyTorch-based Long Short-Term Memory (LSTM) model to forecast the next 7 days of bookings for each city and service category.

## Business Problem

Home service marketplaces experience fluctuations in customer bookings based on:

- Service categories
- City-wise demand
- Day of the week
- Seasonal trends
- Booking time
- Weather conditions

Accurate demand forecasting can help businesses optimize service-provider allocation, workforce planning, and operational efficiency.

## Project Objectives

- Analyze 12 months of historical booking data.
- Perform data cleaning and exploratory data analysis.
- Engineer lag, rolling-window, and calendar features.
- Develop an LSTM-based time-series forecasting model using PyTorch.
- Predict 7-day booking demand for each city and service category.
- Evaluate model performance using appropriate forecasting metrics.
- Deploy predictions through a Flask REST API.
- Automate weekly model retraining.
- Implement basic data drift monitoring and alerts.

## Research and Literature Review

The following three resources provide the theoretical and practical foundation for this project.

### 1. Long Short-Term Memory (LSTM)

**Authors:** Sepp Hochreiter and Jürgen Schmidhuber (1997)

**Research:** Long Short-Term Memory, Neural Computation.

**Link:** https://doi.org/10.1162/neco.1997.9.8.1735

**Key Learning:**
- LSTM is a type of recurrent neural network.
- It uses input, forget, and output gates.
- It captures long-term dependencies in sequential data.
- It is suitable for learning historical demand patterns.

**Application:** LSTM will be used to learn temporal patterns in historical booking data.

### 2. DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks

**Authors:** David Salinas, Valentin Flunkert, Jan Gasthaus and Tim Januschowski.

**Research:** DeepAR – Probabilistic Forecasting with Autoregressive Recurrent Networks.

**Link:** https://arxiv.org/abs/1704.04110

**Key Learning:**
- Uses recurrent neural networks for forecasting multiple related time series.
- Learns patterns across different time series.
- Uses historical observations and additional features.
- Demonstrates the usefulness of global forecasting models.

**Application:** The ideas will help guide forecasting across multiple city and service combinations.

### 3. PyTorch Forecasting – GitHub Repository

**Repository:** PyTorch Forecasting

**Link:** https://github.com/sktime/pytorch-forecasting

**Key Learning:**
- Practical implementation of time-series forecasting models.
- Dataset preparation and feature transformations.
- Forecasting evaluation metrics.
- LSTM and DeepAR model implementations.

**Application:** This repository will be used as a practical reference for model development and evaluation.

## Proposed Model Architecture

The proposed architecture is a multivariate LSTM-based time-series forecasting model.

### Architecture Workflow

```text
Historical Booking Dataset
          |
          v
Data Ingestion
          |
          v
Data Cleaning & Preprocessing
          |
          v
Exploratory Data Analysis
          |
          v
Feature Engineering
          |
          v
Time-Series Dataset Creation
          |
          v
LSTM Model (PyTorch)
          |
          v
Model Training & Validation
          |
          v
Model Evaluation
          |
          v
7-Day Demand Forecast
          |
          v
Flask REST API
          |
          v
Predictions for City + Service
          |
          v
Weekly Retraining & Drift Monitoring
```

### Proposed LSTM Configuration

| Component | Proposed Configuration |
|---|---|
| Framework | PyTorch |
| Model | LSTM |
| Input sequence | Previous 28 days |
| Forecast horizon | Next 7 days |
| Hidden units | 64 |
| LSTM layers | 2 |
| Dropout | 0.2 |
| Optimizer | Adam |
| Loss function | Mean Squared Error |
| Batch size | 32 |
| Learning rate | 0.001 |

*These are initial hyperparameter choices and will be tuned during experimentation.*

### Input Features

| Feature | Description |
|---|---|
| Service Type | Type of home service |
| City | Booking location |
| Date | Booking date |
| Day of Week | Weekly booking pattern |
| Booking Time | Time-related booking information |
| Weather | Weather conditions |
| Lag Features | Previous booking counts |
| Rolling Mean | Average bookings over previous periods |

### Feature Engineering

The following features will be explored:

- Lag 1: Previous day's booking count.
- Lag 7: Booking count from the previous week.
- Lag 14: Booking count from two weeks earlier.
- Rolling Mean 7: Seven-day average booking demand.
- Rolling Mean 14: Fourteen-day average booking demand.
- Day of Week: Monday to Sunday.
- Month: Monthly seasonal patterns.
- Weekend Indicator: Identifies weekends.

All historical features will be calculated using past observations only to prevent data leakage.

Weather information will be included in future predictions only when corresponding weather forecasts are available.

## Model Evaluation Metrics

The model will be evaluated using a chronological four-week hold-out test dataset.

| Metric | Purpose |
|---|---|
| MAPE | Measures percentage forecasting error |
| MAE | Measures average absolute prediction error |
| RMSE | Penalizes larger forecasting errors |
| WAPE | Measures total absolute error relative to total actual demand |

### Acceptance Target

**MAPE ≤ 12%**

The model must achieve a MAPE of 12% or less on the four-week hold-out dataset.

MAPE will be calculated carefully for zero-demand observations; WAPE and MAE will also be reported to provide a more reliable assessment.

## Data Pipeline

The provided dataset will be stored locally in the `data/` directory.

Pipeline stages:

1. Load the booking dataset using Pandas.
2. Inspect column names, data types, missing values, and duplicates.
3. Convert date and time columns into appropriate formats.
4. Aggregate booking counts by date, city, and service category.
5. Handle missing values and outliers.
6. Generate lag and rolling-window features.
7. Encode categorical variables.
8. Scale numerical features using training data only.
9. Prepare sequential input windows for LSTM training.
10. Save processed data for model development.

## Project Structure

```text
urban-company-demand-forecasting/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── api/
│   └── app.py
│
├── scripts/
│   └── weekly_retrain.py
│
├── models/
│
├── reports/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Technology Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deep Learning:** PyTorch
- **Machine Learning:** Scikit-learn
- **API Development:** Flask
- **Automation:** Cron / Scheduled Jobs
- **Version Control:** Git and GitHub

## Deployment and MLOps Plan

The following components are planned for implementation:

1. Flask API to serve 7-day booking forecasts.
2. Weekly automated model retraining using the latest available booking data.
3. Model version management.
4. Monitoring prediction errors and input data distributions.
5. Basic drift detection with alert generation.

## Project Milestones

| Week | Deliverables |
|---|---|
| Week 1 | Research, architecture design, README and data pipeline skeleton |
| Week 2 | Data exploration, cleaning and feature engineering |
| Week 3 | LSTM model development and training |
| Week 4 | Model evaluation and hyperparameter tuning |
| Week 5 | Flask API development and integration |
| Week 6 | Automated retraining and drift monitoring |
| Week 7–12 | Testing, improvements, documentation and final delivery |

## Expected Outcome

A complete demand forecasting pipeline capable of predicting 7-day booking demand for different city and service combinations.

The target is to achieve MAPE of 12% or less on the four-week hold-out dataset and provide predictions through a Flask REST API.

## Dataset

The dataset is provided as part of the project assignment.

**Dataset:** `aiml_urbanclap_bookings.csv`

The dataset should be downloaded and placed inside `data/raw/`.

Note: The dataset is not included in the GitHub repository until its redistribution permissions are confirmed.

## Project Status

**Current Phase:** Week 1 – Research and Architecture Design

**Status:** Initial project setup and architecture planning.

Further implementation and evaluation will be completed in subsequent milestones.
