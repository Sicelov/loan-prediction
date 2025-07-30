# Loan Appproval Prediction

## Table of Contents

- [Overview](#overview)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Data Cleaning](#data-cleaning)
- [Data Analysis](#data-analysis)
- [Key Findings](#key-findings)
- [Conclusion](#conclusion)

### Overview

The Loan Approval Prediction project aims to develop a machine learning model that predicts whether a loan application will be approved based on applicant data. This helps financial institutions automate and enhance the decision-making process, reducing manual effort and bias.

![Loan Application App](https://github.com/Sicelov/loan-prediction/blob/main/Main_Loan.png)

### Data Sources

The dataset used is the Loan Prediction Dataset from Kaggle, which includes:

- Applicant's income
- Coapplicant's income
- Credit history
- Loan amount and term
- Property area
- Education level
- Marital status
- Gender
- Loan status (target: Y/N)

### Tools

- Language: Python
- Data Handling: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-learn (Logistic Regression, Random Forest, XGBoost)
- Model Explainability: SHAP
- Deployment: Streamlit
- Version Control: Git & GitHub
- Notebook Environment: Jupyter Notebook or Google Colab

### Data Cleaning

- Missing Values:
  - Imputed categorical features with mode (e.g., Gender, Marital Status)
  - Imputed numerical features with median (e.g., LoanAmount)

- Encoding:
  - Label encoding for binary categorical variables
  - One-hot encoding for multi-class variables (e.g., Property Area)

- Feature Engineering:
  - Created Total_Income = ApplicantIncome + CoapplicantIncome
  - Applied log transformation to skewed features like LoanAmount

- Outlier Handling:
  - Capped outliers in LoanAmount and Total_Income

### Data Analysis

Most loan approvals are given to applicants with:

  - Good credit history (value = 1)
  - Higher combined income
  - Lower loan amount requests
  - Graduate education level

Property area and marital status had less impact compared to credit history

### Key Findings

- Credit History was the most influential factor in loan approval
- Applicants with higher income and lower loan amounts were more likely to be approved
- Logistic Regression performed well, but XGBoost gave the highest accuracy (~81%) and best recall
- SHAP analysis showed Credit History and Total Income were key drivers of predictions



### Conclusion

The machine learning model accurately predicts loan approvals with high precision and recall using applicant features. This solution can be integrated into banking workflows to assist loan officers and improve turnaround times. Future improvements could involve model retraining with new data and incorporating additional behavioral features (e.g., repayment history, employment verification)


