import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score, RocCurveDisplay,
                             precision_recall_curve)
from xgboost import XGBClassifier

# --------------- Data Cleaning ---------------

# Load dataset
df = pd.read_csv('data/heart_2020_uncleaned.csv')

# Standardize columns
df.columns = df.columns.str.strip().str.replace(' ', '')

# Target to binary
df['HeartDisease'] = df['HeartDisease'].map({'Yes': 1, 'No': 0})

# Lowercase categorical columns
df = df.apply(lambda x: x.str.strip().str.lower() if x.dtype == 'object' else x)

# Define features
target = 'HeartDisease'
num_feats = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
cat_feats = [
    'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex',
    'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity',
    'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer'
]

# Imbalance ratio for scale_pos_weight
imbalance_ratio = (df[target] == 0).sum() / (df[target] == 1).sum()
print(f"Imbalance Ratio (Neg:Pos) = {imbalance_ratio:.2f}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df[num_feats + cat_feats], df[target],
    test_size=0.2, stratify=df[target], random_state=42
)

# Pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preproc = ColumnTransformer([
    ('num', numeric_pipeline, num_feats),
    ('cat', categorical_pipeline, cat_feats)
])

# XGBoost with imbalance handling
model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=imbalance_ratio
)

pipe = Pipeline([
    ('pre', preproc),
    ('clf', model)
])

# Cross-validation
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
print(f'CV ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}')

# Train
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title('ROC Curve – Heart Risk Model')
plt.show()

# --------------- Threshold Adjustment ---------------

prec, rec, thresholds = precision_recall_curve(y_test, y_proba)
desired_recall = 0.30
idx = np.argmax(rec >= desired_recall)
custom_threshold = thresholds[idx]
print(f"Using threshold: {custom_threshold:.2f} for higher sensitivity")

y_pred_custom = (y_proba >= custom_threshold).astype(int)
print("\n🔧 Adjusted Threshold Evaluation:")
print(classification_report(y_test, y_pred_custom))

# Save model
joblib.dump(pipe, 'model/best_heart_disease_model.joblib')
print("Model saved to 'best_heart_disease_model.joblib'")

# Save metadata
all_feats = num_feats + cat_feats

meta = {
    "num_feats": num_feats,
    "cat_feats": cat_feats,
    "all_feats": all_feats,  
    "target": target
}

json.dump(meta, open('model/feature_metadata.json', 'w'))
print("Feature metadata saved to 'feature_metadata.json'")
