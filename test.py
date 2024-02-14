import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import  StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

#load the data
df = pd.read_csv('ACME-HappinessSurvey2020.csv')

#drop least important features
df = df.drop(['X2','X3','X4','X6'], axis=1)

#split into features and target, then split into train and test
X = df.drop('Y', axis=1)
y = df['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#apply SMOTE to balance the data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

#stacking classifier, using Random Forest and XGBoost as base models
base_models = [

    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', xgb.XGBClassifier())
]

#final estimator is Logistic Regression
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(C=1), cv=5)

#fit the model and make predictions
stacking_clf.fit(X_train_smote, y_train_smote)
y_pred = stacking_clf.predict(X_test_scaled)

#evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))