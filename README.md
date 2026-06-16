# Credit-Risk-Analysis
A Python-based data analytics project demonstrating data loading, cleaning, exploratory data analysis (EDA), feature engineering, and database integration to transform raw data into meaningful insights.
# Credit Risk Analysis using Python

## Overview

This project focuses on predicting the probability of loan default and estimating expected financial loss using machine learning techniques. A Logistic Regression model was developed to assess credit risk based on borrowers' financial and credit-related information.

The objective is to support data-driven lending decisions by identifying high-risk borrowers and quantifying potential losses.

---

## Dataset

The dataset contains borrower information, including:

* Credit lines outstanding
* Loan amount outstanding
* Total debt outstanding
* Annual income
* Years employed
* FICO credit score
* Loan default status (Target Variable)

---

## Tools & Libraries Used

* **Python**
* **Pandas** – Data manipulation and analysis
* **Scikit-learn** – Machine learning model development
* **Logistic Regression** – Default probability prediction
* **StandardScaler** – Feature scaling and preprocessing

---

## Project Workflow

### 1. Data Loading

* Imported the loan dataset using Pandas.
* Examined the dataset structure and quality.

### 2. Data Preprocessing

* Selected relevant features influencing credit risk.
* Split the dataset into training and testing sets.
* Standardized numerical features using `StandardScaler`.

### 3. Model Development

* Built a **Logistic Regression** model to predict loan default probability.
* Trained the model using historical borrower data.

### 4. Model Evaluation

* Assessed model performance using classification metrics such as:

  * Precision
  * Recall
  * F1-Score
  * Accuracy

### 5. Expected Loss Estimation

Developed a function to estimate expected financial loss using the formula:

```text
Expected Loss = Probability of Default × Loan Amount × (1 − Recovery Rate)
```

This enables lenders to quantify potential losses associated with individual borrowers.

### 6. Credit Risk Assessment

Tested the model on different borrower profiles to evaluate:

* High-risk borrowers
* Low-risk borrowers
* Average-risk borrowers

---

## Key Features

* Predicts the **Probability of Default (PD)** for loan applicants.
* Estimates **Expected Loss (EL)** using predicted default probabilities.
* Supports **risk-based lending decisions**.
* Demonstrates an end-to-end **credit risk modeling workflow**.

---

## Project Structure

```text
├── task3.py                         # Credit risk analysis script
├── Task 3 and 4_Loan_Data.xlsx      # Loan dataset
├── README.md                        # Project documentation
```

---

## How to Run the Project

### Prerequisites

Install the required libraries:

```bash
pip install pandas scikit-learn openpyxl
```

### Steps

1. Clone this repository:

```bash
git clone https://github.com/your-username/credit-risk-analysis.git
```

2. Navigate to the project directory:

```bash
cd credit-risk-analysis
```

3. Ensure the dataset file is available in the project folder.

4. Run the Python script:

```bash
python task3.py
```

5. Review the model performance metrics and expected loss estimates generated in the console output.

---

## Results

* Successfully developed a Logistic Regression model for credit default prediction.
* Estimated borrower-specific probabilities of default.
* Calculated expected losses to support informed credit risk decisions.
* Demonstrated practical application of machine learning in financial risk management.

---

## Future Improvements

* Experiment with advanced models such as Random Forest, XGBoost, and Gradient Boosting.
* Address class imbalance using resampling techniques.
* Implement cross-validation and hyperparameter tuning.
* Build an interactive dashboard for credit risk monitoring.

---

## Author

**Satyam**

Aspiring Data Analyst | Python | Machine Learning | Credit Risk Analytics

---

*This project demonstrates practical experience in credit risk modeling, probability of default estimation, expected loss calculation, and applying machine learning techniques to real-world financial problems.*
