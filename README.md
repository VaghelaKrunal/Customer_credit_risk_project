# Holistic Data Preparer (Final Project)

## 🎯 Project Objective
The purpose of this project is to test a complete knowledge of Data Preprocessing and Feature Engineering[cite: 2]. The pipeline follows an end-to-end process of preparing a dataset for machine learning: from data understanding to data cleaning, imputation, outlier handling, encoding, scaling, transformations, and feature engineering[cite: 2]. The ultimate outcome is a fully processed dataset ready for ML modeling[cite: 2].

## 💼 Problem Statement
As a Junior Data Scientist at a fintech company, the task is to handle a Customer Credit Risk dataset collected from multiple heterogeneous sources: CSV files, JSON files, SQL tables, and an external API[cite: 2]. The objective is to perform full-scale preprocessing and feature engineering so that the dataset becomes clean, consistent, and perfectly suitable for building a Machine Learning model that predicts whether a customer is likely to default on a loan[cite: 2].

---

## 📊 Dataset Structure & Schema
The dataset contains a rich mix of demographics, financial details, and behavioral attributes[cite: 2]:

* **customer_id** (String / Int): Unique identifier for each customer with no missing values[cite: 2].
* **age** (Integer): Age of customer in years, containing injected missing values for imputation tasks[cite: 2].
* **gender** (Categorical): Gender choices (Male / Female / Other) with missing values and category imbalance[cite: 2].
* **region** (Categorical): Region of residence (North / South / East / West), ideal for One-Hot Encoding[cite: 2].
* **education_level** (Ordinal Cat.): Education background (Primary / Secondary / Graduate / Post-Graduate), tailored for Ordinal Encoding[cite: 2].
* **employment_type** (Categorical): Employment status (Salaried / Self-Employed / Unemployed), containing missing values for categorical imputation[cite: 2].
* **annual_income** (Float): Annual income (₹) exhibiting outliers (very high incomes) and missing values[cite: 2].
* **loan_amount** (Float): Loan amount requested (₹) with outliers and a skewed distribution[cite: 2].
* **loan_purpose** (Categorical): Purpose of loan (Home / Car / Education / Business / Other), designed for One-Hot Encoding[cite: 2].
* **credit_score** (Float): Credit score ranging from 300 to 850, containing outliers and missing values[cite: 2].
* **repayment_history** (Integer): Number of missed payments in the last 12 months, used for Binning and Outlier treatment[cite: 2].
* **transaction_count** (Integer): Total number of transactions in the last 6 months, utilized for K-Means Binning[cite: 2].
* **spending_ratio** (Float): Spending-to-Income ratio (%) used to test Log, Box-Cox, and Yeo-Johnson transforms[cite: 2].
* **join_date** (Date): Date the customer joined the bank, used for handling date/time variables (extracting Year/Month/Day/Weekday)[cite: 2].
* **default_flag** (Binary Int): Target variable where 0 represents No Default and 1 represents Default[cite: 2].

---

## 🛠️ Implementation Pipeline & Tasks

### Part A: Conceptual Foundation
* Explored basic principles of Data Analysis, planning data science projects, and framing machine learning problems[cite: 2].
* Provided an in-depth explanation of Tensors along with practical NumPy examples[cite: 2].

### Part B: Data Acquisition
* Developed modules to import datasets from multiple distinct sources[cite: 2]:
  * Loaded core transactions dataset from CSV files[cite: 2].
  * Parsed customer metadata from JSON files[cite: 2].
  * Fetched loan repayment histories from an SQL database[cite: 2].
  * Fetched real-time external economic indicators from a dummy API[cite: 2].

### Part C: Data Understanding & Cleaning
* Explored the combined dataset using Pandas methods like `.info()` and `.describe()`[cite: 2].
* Generated a detailed data quality report using Pandas Profiling[cite: 2].
* Implemented multi-strategy missing value handling[cite: 2]:
  * Simple Imputation using mean/median for numerical attributes (`age`, `annual_income`, `credit_score`, `employment_type`)[cite: 2].
  * Simple Imputation using most frequent category for categorical attributes[cite: 2].
  * Most Frequent Category Imputation specifically for `gender`[cite: 2].
  * Added Missing Indicators combined with Random Sample Imputation for `annual_income`[cite: 2].
  * Executed advanced multivariate cleaning using KNN Imputer and MICE Algorithm across `annual_income`, `loan_amount`, and `credit_score`[cite: 2].
  * Applied Complete Case Analysis by selectively dropping highly empty rows/columns[cite: 2].

### Part D: Outlier Handling
* Detected and mitigated heavy-tailed outliers across `annual_income`, `loan_amount`, and `credit_score` using four statistical methodologies[cite: 2]:
  * Z-score Method[cite: 2].
  * IQR (Interquartile Range) Method[cite: 2].
  * Percentile Capping Method[cite: 2].
  * Winsorization Technique[cite: 2].

### Part E: Feature Engineering
* Handled mixed variable types containing both numeric and categorical structures (`gender`, `age`, `spending_ratio`)[cite: 2].
* Engineered temporal features from `join_date` by extracting Year, Month, Day, and Weekday[cite: 2].
* Applied tailored categorical encoding workflows[cite: 2]:
  * Ordinal Encoding for `education_level`[cite: 2].
  * Label Encoding for binary features like `gender`[cite: 2].
  * One-Hot Encoding for multi-class variables like `region` and `loan_purpose`[cite: 2].
* Conducted numerical encoding, discretization, and advanced binning[cite: 2]:
  * Numerical encoding for `repayment_history` and `transaction_count`[cite: 2].
  * Discretized and binned `annual_income` and `repayment_history` into distinct income groups[cite: 2].
  * Binarized credit data by flagging instances where `credit_score > 700`[cite: 2].
  * Applied Quantile and K-Means Binning on `transaction_count`[cite: 2].

### Part F: Feature Scaling
* Evaluated and applied multiple feature scaling methodologies across all numeric columns[cite: 2]:
  * Standardization (Z-score scaling) applied specifically to `annual_income` and `loan_amount`[cite: 2].
  * Normalization[cite: 2].
  * Min-Max Scaling[cite: 2].
  * MaxAbs Scaling[cite: 2].
  * Robust Scaling[cite: 2].

### Part G: Feature Construction & Transformation
* Addressed non-normal distributions and skewness using mathematical transformations[cite: 2]:
  * Applied `FunctionTransformer` to perform log, reciprocal, and square root transforms on `spending_ratio`[cite: 2].
  * Utilized `PowerTransformer` to apply Box-Cox and Yeo-Johnson algorithms on skewed fields like `loan_amount` and `annual_income`[cite: 2].
* Engineered brand-new structural business features[cite: 2]:
  * Constructed the **Debt-to-Income ratio** (`loan_amount / annual_income`)[cite: 2].
  * Derived average monthly transactions[cite: 2].
  * Computed the spending-to-income ratio[cite: 2].
* Integrated the entire preprocessing workflow into an isolated, leakage-free scikit-learn `ColumnTransformer` pipeline to apply different steps to categorical vs numeric features seamlessly[cite: 2].

---

## 🏆 Final Deliverables & Expected Outcome
By the end of this project, the pipeline generates[cite: 2]:
1. A finalized, completely cleaned, and engineered dataset ready for direct input into predictive Machine Learning models[cite: 2].
2. A comprehensive analytical report summarizing the effectiveness of missing value strategies, outlier capping results, applied encoding/scaling methods, and the predictive utility of newly constructed features[cite: 2].
