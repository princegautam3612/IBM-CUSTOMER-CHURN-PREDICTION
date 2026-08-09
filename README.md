# 📉 Telco Customer Churn Predictor

**Predicting which customers are about to leave — before they do.**

An end-to-end machine learning project that identifies at-risk telecom customers, explains *why* they're at risk, and delivers it through an interactive app. Built on IBM's Telco Customer Churn dataset (~7,000 customers).

> Catches **78% of customers who actually churn**, with a live app that explains every prediction and surfaces the underlying data patterns.

---

## Why This Matters

Acquiring a new customer costs far more than retaining an existing one. Most churn happens quietly — a customer doesn't complain, they just leave. The goal of this project isn't just "predict churn," it's to catch that customer *early enough for a business to act*, and to explain *why* they're at risk so retention efforts are targeted, not generic.

## What This Project Does

- 🔍 **Digs into the data** — uncovers non-obvious churn patterns (like a U-shaped tenure curve: new customers and long-timers behave completely differently)
- ⚖️ **Compares 5 models head-to-head** — Logistic Regression, KNN, SVM, Random Forest, XGBoost — with hyperparameter tuning, not just defaults
- 🎯 **Optimizes for the right metric** — recall over raw accuracy, because missing a churner costs more than a false alarm
- 🧠 **Explains every prediction** — SHAP values show exactly which factors are pushing a specific customer toward the door
- 🖥️ **Ships as a working app** — a Streamlit interface with both a live predictor and a data insights dashboard, not just a notebook

---

## Key Insights from the Data

| Insight | What it means |
|---|---|
| **Tenure is U-shaped** | Customers churn heavily in their first year, or stay for 5+ years — almost nobody churns in the middle. Retention efforts should focus hardest on the first 12 months. |
| **Contract type is the #1 driver** | Month-to-month customers churn dramatically more than annual/biennial contract holders — the single strongest lever a business has. |
| **Fiber optic + electronic check = red flag combo** | This customer segment shows elevated churn, worth investigating pricing or service quality. |
| **`Churn Reason` confirmed real drivers** | Top stated reasons: competitor offers, service attitude, and pricing — validated against the model's own top features. |

Two columns were **deliberately excluded** from modeling to avoid data leakage: `Churn Reason` (only known *after* a customer leaves) and `Churn Score` (IBM's own precomputed risk score — including it would let the model cheat off an answer key).

---

## Model Performance

| Model | Accuracy | Recall (churn) | Precision (churn) |
|---|---|---|---|
| Logistic Regression | 75.6% | 0.54 | 0.81 |
| KNN | 78% | 0.55 | 0.75 |
| SVM | 78% | 0.55 | 0.61 |
| XGBoost (tuned) | 75% | 0.83 | 0.54 |
| **🏆 Random Forest (tuned)** | **78%** | **0.78** | **0.59** |

**Random Forest won** — not because it topped every column, but because it struck the best real-world balance. A model optimized purely for accuracy would miss nearly half of actual churners; this one catches 78% of them.

<details>
<summary>Final hyperparameters (found via RandomizedSearchCV, scoring='f1')</summary>

```python
RandomForestClassifier(
    n_estimators=200, min_samples_split=2, min_samples_leaf=1,
    max_features='sqrt', max_depth=10, bootstrap=False,
    class_weight='balanced'
)
```
</details>

---

## The App

A two-tab Streamlit interface:

**🔮 Predict** — fill in a customer's profile and get an instant churn probability, plus a SHAP explanation chart showing the top factors pushing that specific customer toward or away from churn.

**📊 Data Insights** — a live dashboard of the dataset: churn rate by tenure and contract type, monthly charges by churn status, top stated reasons customers left, and the model's own feature importance ranking — tying the analysis directly back to what the model learned.



---

## Project Structure

```
telco-churn-predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── Churn.py               # full EDA + model comparison (Logistic Regression, KNN, SVM, RF, XGBoost)
├── train_pipeline.py      # builds the final preprocessing + model pipeline, saves churn_pipeline.pkl
├── app.py                 # Streamlit app: live prediction + data insights dashboard
└── churn_pipeline.pkl     # trained model (or regenerate via train_pipeline.py)
```

`Churn.py` is the exploration script — EDA, correlation analysis, and comparison across five models. `train_pipeline.py` is the final, consolidated pipeline built from those findings, bundling preprocessing and the tuned Random Forest into a single `sklearn.Pipeline` object.

## Run It Yourself

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset (link below) and place Telco_customer_churn.xlsx in the project root

# 3. (Optional) Explore the EDA and model comparison
python Churn.py

# 4. Train the final model
python train_pipeline.py

# 5. Launch the app
streamlit run app.py
```

**Dataset:** [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Tech Stack

`pandas` `scikit-learn` `XGBoost` `SHAP` `Streamlit` `matplotlib` `seaborn`

## What's Next

- 📤 Bulk prediction via CSV upload for scoring entire customer lists
- 💡 Rule-based retention suggestions tied to each customer's top churn drivers
- ☁️ Public deployment on Streamlit Community Cloud

---

*Built by Prince Gautam —  https://github.com/princegautam3612 · www.linkedin.com/in/prince-gautam-623598384
