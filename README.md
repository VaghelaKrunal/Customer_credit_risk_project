# Holistic Data Preparer (Final Project)

## 🎯 Project Objective
The purpose of this project is to test a complete knowledge of Data Preprocessing and Feature Engineering. The pipeline follows an end-to-end process of preparing a dataset for machine learning: from data understanding to data cleaning, imputation, outlier handling, encoding, scaling, transformations, and feature engineering. The ultimate outcome is a fully processed dataset ready for ML modeling.

## 💼 Problem Statement
As a Junior Data Scientist at a fintech company, the task is to handle a Customer Credit Risk dataset collected from multiple heterogeneous sources: CSV files, JSON files, SQL tables, and an external API. The objective is to perform full-scale preprocessing and feature engineering so that the dataset becomes clean, consistent, and perfectly suitable for building a Machine Learning model that predicts whether a customer is likely to default on a loan.

---

## 📊 Dataset Structure & Schema
The dataset contains a rich mix of demographics, financial details, and behavioral attributes:

* **customer_id** (String / Int): Unique identifier for each customer with no missing values.
* **age** (Integer): Age of customer in years, containing injected missing values for imputation tasks.
* **gender** (Categorical): Gender choices (Male / Female / Other) with missing values and category imbalance.
* **region** (Categorical): Region of residence (North / South / East / West), ideal for One-Hot Encoding.
* **education_level** (Ordinal Cat.): Education background (Primary / Secondary / Graduate / Post-Graduate), tailored for Ordinal Encoding.
* **employment_type** (Categorical): Employment status (Salaried / Self-Employed / Unemployed), containing missing values for categorical imputation.
* **annual_income** (Float): Annual income (₹) exhibiting outliers (very high incomes) and missing values.
* **loan_amount** (Float): Loan amount requested (₹) with outliers and a skewed distribution.
* **loan_purpose** (Categorical): Purpose of loan (Home / Car / Education / Business / Other), designed for One-Hot Encoding.
* **credit_score** (Float): Credit score ranging from 300 to 850, containing outliers and missing values.
* **repayment_history** (Integer): Number of missed payments in the last 12 months, used for Binning and Outlier treatment.
* **transaction_count** (Integer): Total number of transactions in the last 6 months, utilized for K-Means Binning.
* **spending_ratio** (Float): Spending-to-Income ratio (%) used to test Log, Box-Cox, and Yeo-Johnson transforms.
* **join_date** (Date): Date the customer joined the bank, used for handling date/time variables (extracting Year/Month/Day/Weekday).
* **default_flag** (Binary Int): Target variable where 0 represents No Default and 1 represents Default.

---

## 🛠️ Implementation Pipeline & Tasks

### Part A: Conceptual Foundation
* Explored basic principles of Data Analysis, planning data science projects, and framing machine learning problems.
* Provided an in-depth explanation of Tensors along with practical NumPy examples.

### Part B: Data Acquisition
* Developed modules to import datasets from multiple distinct sources:
  * Loaded core transactions dataset from CSV files.
  * Parsed customer metadata from JSON files.
  * Fetched loan repayment histories from an SQL database.
  * Fetched real-time external economic indicators from a dummy API.

### Part C: Data Understanding & Cleaning
* Explored the combined dataset using Pandas methods like `.info()` and `.describe()`.
* Generated a detailed data quality report using Pandas Profiling.
* Implemented multi-strategy missing value handling:
  * Simple Imputation using mean/median for numerical attributes (`age`, `annual_income`, `credit_score`, `employment_type`).
  * Simple Imputation using most frequent category for categorical attributes.
  * Most Frequent Category Imputation specifically for `gender`.
  * Added Missing Indicators combined with Random Sample Imputation for `annual_income`.
  * Executed advanced multivariate cleaning using KNN Imputer and MICE Algorithm across `annual_income`, `loan_amount`, and `credit_score`.
  * Applied Complete Case Analysis by selectively dropping highly empty rows/columns.

### Part D: Outlier Handling
* Detected and mitigated heavy-tailed outliers across `annual_income`, `loan_amount`, and `credit_score` using four statistical methodologies:
  * Z-score Method.
  * IQR (Interquartile Range) Method.
  * Percentile Capping Method.
  * Winsorization Technique.

### Part E: Feature Engineering
* Handled mixed variable types containing both numeric and categorical structures (`gender`, `age`, `spending_ratio`).
* Engineered temporal features from `join_date` by extracting Year, Month, Day, and Weekday.
* Applied tailored categorical encoding workflows:
  * Ordinal Encoding for `education_level`.
  * Label Encoding for binary features like `gender`.
  * One-Hot Encoding for multi-class variables like `region` and `loan_purpose`.
* Conducted numerical encoding, discretization, and advanced binning:
  * Numerical encoding for `repayment_history` and `transaction_count`.
  * Discretized and binned `annual_income` and `repayment_history` into distinct income groups.
  * Binarized credit data by flagging instances where `credit_score > 700`.
  * Applied Quantile and K-Means Binning on `transaction_count`.

### Part F: Feature Scaling
* Evaluated and applied multiple feature scaling methodologies across all numeric columns:
  * Standardization (Z-score scaling) applied specifically to `annual_income` and `loan_amount`.
  * Normalization.
  * Min-Max Scaling.
  * MaxAbs Scaling.
  * Robust Scaling.

### Part G: Feature Construction & Transformation
* Addressed non-normal distributions and skewness using mathematical transformations:
  * Applied `FunctionTransformer` to perform log, reciprocal, and square root transforms on `spending_ratio`.
  * Utilized `PowerTransformer` to apply Box-Cox and Yeo-Johnson algorithms on skewed fields like `loan_amount` and `annual_income`.
* Engineered brand-new structural business features:
  * Constructed the **Debt-to-Income ratio** (`loan_amount / annual_income`).
  * Derived average monthly transactions.
  * Computed the spending-to-income ratio.
* Integrated the entire preprocessing workflow into an isolated, leakage-free scikit-learn `ColumnTransformer` pipeline to apply different steps to categorical vs numeric features seamlessly.

---

## 🏆 Final Deliverables & Expected Outcome
By the end of this project, the pipeline generates:
1. A finalized, completely cleaned, and engineered dataset ready for direct input into predictive Machine Learning models.
2. A comprehensive analytical report summarizing the effectiveness of missing value strategies, outlier capping results, applied encoding/scaling methods, and the predictive utility of newly constructed features.
