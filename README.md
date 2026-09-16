# Bias and Fairness Audit of an Income Prediction Model

This project looks at whether an income prediction model treats different groups of people fairly. I used the UCI Adult Income dataset to build a model, check for bias, and then apply a simple fairness fix to see how it changes the results.

## About the dataset

The project uses the **UCI Adult Income dataset**, which contains US Census data from 1994. It has around **45,000 records**.

Each record includes information like:

- Age
- Education
- Occupation
- Hours worked per week
- Marital status
- Sex
- Race

The goal is to predict whether a person earns **more than $50,000 per year**.

## What I did

### 1. Cleaned the data

- Removed rows with missing values.
- Cleaned text formatting.
- Converted the income column into a binary value:
  - `0` = income is **$50K or less**
  - `1` = income is **more than $50K**

### 2. Trained a baseline model

I trained a **Logistic Regression** model without making any fairness changes.

**Accuracy:** **81.8%**

This is the baseline model that is used for comparison.

### 3. Checked for bias

I tested the model separately for different groups based on **sex** and **race**.

I used two fairness measures:

- **Selection Rate** – Percentage of people predicted as earning more than $50K.
- **True Positive Rate** – Percentage of actual high earners that the model correctly predicted.

### Results

**Selection Rate**

- Men predicted as high income: **21.3%**
- Women predicted as high income: **3.3%**

Gap: **18 percentage points**

**True Positive Rate**

- Men: **49%**
- Women: **19.5%**

The model was much better at identifying high-income men than high-income women.

### 4. Found the reason for the bias

I checked if the model could still guess someone's sex after removing the `sex` column.

It could.

The `relationship` column contains values like **Husband**, **Wife**, **Unmarried**, and **Own-child**. This column is strongly connected to a person's sex, so the model can still learn that information.

This is called a **proxy variable**. It indirectly reveals a protected attribute, even when that attribute is removed.

This was the biggest learning from the project because it shows why simply deleting a sensitive column is not enough.

### 5. Reduced the bias

I used **threshold adjustment**.

Instead of using the same prediction cutoff for everyone, I used different cutoffs for men and women so that both groups had a similar selection rate.

### Results after the fix

| Metric | Before | After |
|--------|--------|-------|
| Accuracy | **81.8%** | **79.4%** |
| Selection rate gap | **18.0%** | **0.1%** |

The fairness gap became much smaller, but the model lost **2.4% accuracy**.

This shows that improving fairness can come with a small drop in overall performance.

## Project files

- **fairness_audit.py** – Cleans the data, trains the model, checks fairness, applies the fairness fix, and saves the results.
- **app.py** – Streamlit dashboard that shows the audit results and lets you try predictions with your own input.
- **adult.csv** – Dataset used in the project.
- **results_summary.json** – Stores the results used by the dashboard.

## How to run the project

### Install the required libraries

```bash
pip install pandas numpy scikit-learn shap streamlit --break-system-packages
```

### Run the fairness audit

```bash
python3 fairness_audit.py
```

### Open the dashboard

```bash
streamlit run app.py
```

## What I learned

- A model can be accurate but still treat some groups unfairly.
- Removing a sensitive feature does not always remove bias because of proxy variables.
- Fairness can be improved, but it may reduce accuracy.
- Measuring both fairness and accuracy helps understand the trade-off before using a model in a real application.
