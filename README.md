
# 🏦 Credit Risk Classification Project

This project predicts **credit risk (Good or Bad)** using the **German Credit Risk dataset**.
It involves **data preprocessing, exploratory data analysis (EDA)**, **machine learning model training**, and a **Streamlit web app** for deployment.

---

## 📂 Project Overview

The goal is to build a machine learning model that classifies customers into **Good (low risk)** or **Bad (high risk)** credit risk categories based on their financial and personal attributes.

---

## 📊 Dataset Description

**Dataset name:** `german_credit_data.csv`
**Source:** UCI German Credit Risk Dataset

### Columns:

| Feature          | Description                              |
| ---------------- | ---------------------------------------- |
| Age              | Age of the applicant                     |
| Sex              | Gender of the applicant                  |
| Job              | Job level (0–3)                          |
| Housing          | Type of housing (own, rent, free)        |
| Saving accounts  | Savings account balance level            |
| Checking account | Checking account balance level           |
| Credit amount    | Amount of credit requested               |
| Duration         | Duration of the credit in months         |
| Purpose          | Purpose of the loan                      |
| Risk             | Target variable (Good / Bad credit risk) |

---

## ⚙️ Steps Followed

### 1️⃣ Import Dependencies

Imported libraries like **pandas**, **numpy**, **matplotlib**, **seaborn**, **scikit-learn**, **xgboost**, and **joblib**.

### 2️⃣ Data Loading & Cleaning

* Loaded dataset using `pandas.read_csv()`.
* Removed duplicate records.
* Dropped missing values.
* Dropped unnecessary column `Unnamed: 0`.

### 3️⃣ Exploratory Data Analysis (EDA)

* Visualized numerical and categorical feature distributions.
* Compared “Good” vs “Bad” risk profiles.
* Generated correlation heatmaps and boxplots.
* Explored relationships between **Age**, **Credit amount**, **Duration**, and **Risk**.

### 4️⃣ Feature Engineering

* Identified **numerical** and **categorical** columns.
* Encoded categorical variables using **LabelEncoder**.
* Saved encoders as `.pkl` files using **joblib** for deployment.

### 5️⃣ Model Building

Split dataset into **train** and **test** sets using:

```python
train_test_split(x, y, test_size=0.2, random_state=1)
```

Trained multiple classifiers with **GridSearchCV** for hyperparameter tuning:

* **Decision Tree**
* **Random Forest**
* **Extra Trees**
* **XGBoost**

Each model was evaluated using **accuracy score**.

---

## 🧠 Model Performance

| Model         | Accuracy   | Best Parameters                                                                                          |
| ------------- | ---------- | -------------------------------------------------------------------------------------------------------- |
| Decision Tree | 52.38%     | `{'max_depth': None, 'min_samples_leaf': 4, 'min_samples_split': 10}`                                    |
| Extra Trees   | 49.52%     | `{'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}`                |
| XGBoost       | **58.09%** | `{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.7}` |

✅ **Best Model:** XGBoost Classifier
Saved as `xgb_credit_model.pkl`.

---

## 💾 Model Saving

Encoders and model were saved for later use in the Streamlit app:

```python
joblib.dump(best_xgb, "xgb_credit_model.pkl")
joblib.dump(le, f"{col}_encoder.pkl")
joblib.dump(le_target, "target_encoder.pkl")
```

---

## 🌐 Streamlit Web App

A user-friendly **Streamlit application** was built to interact with the trained model.

### Features:

* Input fields for user data (Age, Sex, Job, etc.)
* Encoded categorical variables automatically.
* Displays prediction result as:

  * ✅ **GOOD** (Low Risk)
  * ❌ **BAD** (High Risk)

### How to Run:

```bash
streamlit run app.py
```

---

## 🧩 Project Structure

```
📁 German-Credit-Risk/
│
├── german_credit_data.csv
├── credit_model_training.ipynb
├── app.py                     # Streamlit app
├── xgb_credit_model.pkl       # Trained model
├── target_encoder.pkl
├── Sex_encoder.pkl
├── Housing_encoder.pkl
├── Saving accounts_encoder.pkl
├── Checking account_encoder.pkl
└── README.md
```

---

## 🚀 How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/credit-risk-classification.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## 🧮 Technologies Used

* **Python**
* **Pandas, NumPy**
* **Matplotlib, Seaborn**
* **Scikit-learn**
* **XGBoost**
* **Streamlit**
* **Joblib**

---

## 🧭 Conclusion

This project demonstrates:

* A full end-to-end **machine learning workflow**.
* A practical deployment using **Streamlit**.
* How to evaluate and compare different models for **credit risk classification**.

Would you like me to also generate a **`requirements.txt`** file for your GitHub (with all the needed libraries and versions)?
