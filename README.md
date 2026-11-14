
# Customer Subscription Prediction Using Machine Learning

## 1. Project Overview

This project develops a machine learning system that predicts whether a bank customer is likely to subscribe to a term deposit.

The project uses the UCI Bank Marketing dataset and compares Logistic Regression with Random Forest classification.

## 2. Problem Statement

Banks conduct marketing campaigns to encourage customers to subscribe to term deposits.

The objective is to use historical customer and campaign information to estimate the likelihood of subscription and support better targeting of marketing efforts.

## 3. Dataset

Source: UCI Machine Learning Repository

Dataset: Bank Marketing

https://archive.ics.uci.edu/dataset/222/bank+marketing

The bank-full dataset contains 45,211 records and 16 input variables, together with a binary target variable.

The target variable is:

- yes: Customer subscribed.
- no: Customer did not subscribe.

The `duration` variable was excluded because call duration is only known after a marketing call and would introduce information leakage into a pre-call prediction system.

## 4. Machine Learning Models

The project evaluates:

1. Logistic Regression
2. Random Forest Classifier

Random Forest was selected as the final model after model comparison and hyperparameter tuning.

## 5. Final Model

Model: Tuned Random Forest

Classification threshold: 0.60

Final test performance:

| Metric | Score |
|---|---:|
| Accuracy | 0.8805 |
| Precision | 0.4887 |
| Recall | 0.4688 |
| F1-score | 0.4785 |
| ROC-AUC | 0.8003 |
| PR-AUC | 0.4506 |

## 6. API

The project uses FastAPI to provide predictions.

Available endpoints:

- GET `/` - API welcome message.
- GET `/health` - API health status.
- POST `/predict` - Predicts customer subscription.

## 7. Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
