import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# 1. Load and do the MINIMAL cleaning that must happen before
#    the pipeline (things that aren't "transform a column",
#    like fixing dtypes or dropping known-bad rows/columns).
# ---------------------------------------------------------
df = pd.read_excel("Telco_customer_churn.xlsx")

df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df.dropna(subset=['Total Charges'], inplace=True)

df = df.drop(columns=[
    'Churn Label', 'Lat Long', 'CustomerID', 'Churn Reason', 'Churn Score',
    'City', 'State', 'Country', 'Zip Code', 'Latitude', 'Longitude', 'Count'
], errors='ignore')

# Binary yes/no-style columns -> map to 0/1 directly (simpler than LabelEncoder,
# and avoids LabelEncoder's "fit on train, but what about unseen categories" issue)
binary_map_cols = ['Senior Citizen', 'Gender', 'Partner', 'Dependents',
                    'Phone Service', 'Paperless Billing']
binary_maps = {
    'Senior Citizen': {'No': 0, 'Yes': 1, 0: 0, 1: 1},
    'Gender': {'Female': 0, 'Male': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'Phone Service': {'No': 0, 'Yes': 1},
    'Paperless Billing': {'No': 0, 'Yes': 1},
}
for col, mapping in binary_maps.items():
    df[col] = df[col].map(mapping)

# Multi-category columns -> the Pipeline's OneHotEncoder will handle these
multi_cols = ['Multiple Lines', 'Online Security', 'Online Backup', 'Device Protection',
              'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract',
              'Payment Method', 'Internet Service']

numeric_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV']

X = df.drop(columns=['Churn Value'])
y = df['Churn Value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 2. Build the preprocessing + model pipeline
# ---------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), multi_cols),
    ],
    remainder='passthrough'  # keeps the already-binary columns as-is
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=200, min_samples_split=2, min_samples_leaf=1,
        max_features='sqrt', max_depth=10, bootstrap=False,
        class_weight='balanced', random_state=42
    ))
])

# ---------------------------------------------------------
# 3. Fit and evaluate
# ---------------------------------------------------------
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 4. Save the ENTIRE pipeline as one file
# ---------------------------------------------------------
joblib.dump(pipeline, 'churn_pipeline.pkl')
print("Saved churn_pipeline.pkl — preprocessing + model in one object.")
