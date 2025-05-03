import joblib
import numpy as np

# -------------------- Load Model and Encoders --------------------
model = joblib.load('model/hedge_fund_model.pkl')
label_encoder = joblib.load('model/stock_label_encoder.pkl')
features = joblib.load('model/model_features.pkl')  # 🔥 Load feature names (71 total)

# -------------------- Get User Inputs --------------------
print("🔎 Please enter the following company information for prediction:")

stock_name = input("Enter the company stock name (e.g., RELIANCE, TCS): ").strip().upper()

try:
    stock_code = label_encoder.transform([stock_name])[0]
except ValueError:
    print(f"\n❌ Error: '{stock_name}' was not found in the training data.")
    exit()

try:
    net_income = float(input("Enter the Net Income (e.g., 123456.78): "))
except ValueError:
    print("❌ Invalid net income. Please enter a numeric value.")
    exit()

# -------------------- Construct Full Feature Vector --------------------
# Set default 0.0 for all features, then override known ones
input_data = {feature: 0.0 for feature in features}
input_data['stock'] = stock_code
input_data['netIncome_x'] = net_income

# -------------------- Convert to Model Input --------------------
input_array = np.array([[input_data[feature] for feature in features]])

# -------------------- Make Prediction --------------------
prediction = model.predict(input_array)

# -------------------- Output Result --------------------
if prediction[0] == 1:
    print(f"\n✅ Prediction: The company '{stock_name}' is making a **PROFIT**.")
else:
    print(f"\n❌ Prediction: The company '{stock_name}' is making a **LOSS**.")
