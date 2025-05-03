import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib

# ---------------------- Load Data ----------------------
annual_data = pd.read_csv("new_data/new_balance_cashflow_incom_annualmerged.csv")
quarterly_data = pd.read_csv("new_data/new_balance_cashflow_incom_quarterlymerged.csv")

# Combine both datasets
df = pd.concat([annual_data, quarterly_data], ignore_index=True)

# ---------------------- Preprocessing ----------------------

# Convert 'endDate' to datetime
df['endDate'] = pd.to_datetime(df['endDate'], errors='coerce')

# Drop rows where endDate couldn't be parsed (optional)
df.dropna(subset=['endDate'], inplace=True)

# Create target variable: profit (1) or loss (0)
df['profit_loss'] = (df['netIncome_x'] > 0).astype(int)

# Extract date features
df['year'] = df['endDate'].dt.year
df['month'] = df['endDate'].dt.month
df['quarter'] = df['endDate'].dt.quarter

# Label Encode 'stock' (company name)
label_encoder = LabelEncoder()
df['stock'] = label_encoder.fit_transform(df['stock'])

# Drop 'endDate' since it's now encoded into other features
df.drop(columns=['endDate'], inplace=True)

# ---------------------- Feature & Target ----------------------
target = 'profit_loss'
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# ---------------------- Split Data ----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------- Train Model ----------------------
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    max_depth=6,
    n_estimators=100,
    learning_rate=0.1
)

model.fit(X_train, y_train)

# ---------------------- Evaluate Model ----------------------
y_pred = model.predict(X_test)

print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------- Save Model ----------------------
joblib.dump(model, 'model/hedge_fund_model.pkl')
print("\n💾 Model saved as 'hedge_fund_model.pkl'.")
