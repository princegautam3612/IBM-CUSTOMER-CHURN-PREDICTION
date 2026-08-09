import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

df=pd.read_excel("Telco_customer_churn.xlsx")

# Performing EDA
df['Total Charges']=pd.to_numeric(df['Total Charges'],errors='coerce')
df.dropna(subset=['Total Charges'],inplace=True)
df = df.drop(columns=['Churn Label','Lat Long','CustomerID']) 

churn_reasons=df['Churn Reason'].copy()
df=df.drop(columns=['Churn Reason'])

# Encoding(Label Encoding)
encoder=LabelEncoder()
binary_cols=['Senior Citizen','Gender','Partner','Dependents','Phone Service','Paperless Billing']
for col in binary_cols:
    df[col]=encoder.fit_transform(df[col])

# Encoding(One Hot Encoding)
multi_cols=['Multiple Lines','Online Security','Online Backup','Device Protection','Tech Support','Streaming TV','Streaming Movies','Contract','Payment Method','Internet Service']
df=pd.get_dummies(df,columns=multi_cols,drop_first=True)


# Distributions
numeric_cols=['Total Charges','Tenure Months','Monthly Charges']
fig,axes=plt.subplots(2,3,figsize=(18,8))

for i,col in enumerate(numeric_cols):
    sns.histplot(df[col],kde=True,ax=axes[0,i])
    axes[0,i].set_title(f'Distribution of {col}')

    sns.boxplot(x=df[col],ax=axes[1,i])
    axes[1,i].set_title(f'Boxplot of {col}')

plt.tight_layout()

# for 0/1(checking the split )

# for col in binary_cols:
#     print(df[col].value_counts(normalize=True),'\n')

# for one hot encoded columns

dummy_cols=['Multiple Lines_No phone service', 'Multiple Lines_Yes',
    'Online Security_No internet service', 'Online Security_Yes',
    'Online Backup_No internet service', 'Online Backup_Yes',
    'Device Protection_No internet service', 'Device Protection_Yes',
    'Tech Support_No internet service', 'Tech Support_Yes',
    'Streaming TV_No internet service', 'Streaming TV_Yes',
    'Streaming Movies_No internet service', 'Streaming Movies_Yes']

df[dummy_cols].sum().sort_values(ascending=False).plot(kind='bar',figsize=(14,5))

plt.title('counts per one hot category')
plt.ylabel('counts')
plt.tight_layout()

df_eda = df.copy()
df_eda['Tenure Group'] = pd.cut(df_eda['Tenure Months'], bins=[0,12,24,36,48,60,72])
churn_by_tenure = df_eda.groupby('Tenure Group')['Churn Value'].mean()

churn_by_tenure.plot(kind='bar',figsize=(10,6),color='pink')
plt.ylabel("Churn Rate")
plt.title("Churn rate by tenure months") # this gives that new costumer churn more

contact_cols=['Contract_One year', 'Contract_Two year']
for col in contact_cols:
    print(col,"Churn Rate",df[col].mean())

sns.boxplot(x='Churn Value', y='Monthly Charges', data=df)
plt.title('Monthly Charges by Churn Status')

# making correlation heatmap

plt.figure(figsize=(16,13))
sns.heatmap(df.corr(numeric_only=True)[['Churn Value']].sort_values(by='Churn Value',ascending=False),annot=True,cmap='coolwarm')
plt.title('Correlation with Churn')
# plt.show()

# print(df.columns)
df_cleaned=df.drop(columns=['Multiple Lines_No phone service','Online Backup_No internet service','Tech Support_No internet service','Streaming TV_No internet service','Streaming Movies_No internet service','Churn Score','Online Security_No internet service','City','State','Country','Zip Code','Latitude','Longitude','Count','No internet Service'],errors='ignore')
df_cleaned=df_cleaned.rename(columns={'Device Protection_No internet service':'No internet Service'})

bool_cols=df_cleaned.select_dtypes(include='bool').columns
df_cleaned[bool_cols]=df_cleaned[bool_cols].astype(int)

# print(df_cleaned.info())

# Now we are doing standard scaling 

from sklearn.preprocessing import StandardScaler
cols_to_scale=['CLTV','Total Charges','Monthly Charges','Tenure Months']
scaler=StandardScaler()
for col in cols_to_scale:
    df_cleaned[cols_to_scale]=scaler.fit_transform(df_cleaned[cols_to_scale])

# Performing Spliting Data
from sklearn.model_selection import train_test_split
X=df_cleaned.drop(columns=['Churn Value'])
y=df_cleaned['Churn Value']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# Testing Base Model

        # Model 1- Logistic Regression(Accuracy-75.6%,Recall(Churn)=0.54,Precision(churn)=0.81)
from sklearn.linear_model import LogisticRegression
model_lr=LogisticRegression(max_iter=1000,class_weight='balanced')
model_lr.fit(X_train,y_train)
y_predlr=model_lr.predict(X_test)

acclr=accuracy_score(y_predlr,y_test)
reportlr=classification_report(y_predlr,y_test)
matrixlr=confusion_matrix(y_predlr,y_test)
# print(acclr,reportlr,matrixlr)

# sclaing full data for model checks like (KNN and SVM)

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


        # Model 2-KNN(Accuracy-78%,Recall(Churn)=0.55,Precision(churn)=0.75)

from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_scaled, y_train)
y_pred_knn = model_knn.predict(X_test_scaled)
# print(classification_report(y_test, y_pred_knn))

        # Model 3-SVM()
from sklearn.svm import SVC
model_svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
model_svm.fit(X_train_scaled, y_train)
y_pred_svm = model_svm.predict(X_test_scaled)
# # print(classification_report(y_test, y_pred_svm))

# Random Forest

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=10,
    bootstrap=False,
    class_weight='balanced',
    random_state=42
)
model_rf.fit(X_train, y_train)
y_predrf = model_rf.predict(X_test)

accrf=accuracy_score(y_test,y_predrf)
reportrf=classification_report(y_test,y_predrf)
matrixrf=confusion_matrix(y_test,y_predrf)

print(accrf,reportrf,matrixrf)

from xgboost import XGBClassifier
model_xgb=XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=(y_train.value_counts()[0]/y_train.value_counts()[1]),
    random_state=42,
    eval_metric='logloss'
)

model_xgb.fit(X_train,y_train)
y_predxgb=model_xgb.predict(X_test)

accxgb=accuracy_score(y_test,y_predxgb)
reportxgb=classification_report(y_test,y_predxgb)
matrixxgb=confusion_matrix(y_test,y_predxgb)

# print(accxgb,reportxgb,matrixxgb)

                    # HyperParameter Tuning

from sklearn.model_selection import RandomizedSearchCV

Grid = RandomizedSearchCV(
    model_rf,
    {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    },
    n_iter=50, cv=5, random_state=42, n_jobs=-1,scoring='f1'
)
Grid.fit(X_train, y_train)
# print("Best RF params:", Grid.best_params_, "F1 CV score:", Grid.best_score_)
best_rf = Grid.best_estimator_
# print(classification_report(y_test, best_rf.predict(X_test)))


Grid2=RandomizedSearchCV((model_xgb),{
    'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
},n_iter=50, cv=5, random_state=42, n_jobs=-1,scoring='f1')

Grid2.fit(X_train, y_train) 

# print("Best XGB params:", Grid2.best_params_, "F1 CV score:", Grid2.best_score_)
# best_xgb = Grid2.best_estimator_
# print(classification_report(y_test, best_xgb.predict(X_test)))