import streamlit as st
import joblib
import numpy as np

# -------------------- Load Model and Label Encoder --------------------
model = joblib.load('model/hedge_fund_model.pkl')  # Your trained classifier
label_encoder = joblib.load('model/stock_label_encoder.pkl')  # LabelEncoder saved during training
features = joblib.load('model/model_features.pkl')  # Features saved during training

# -------------------- Streamlit User Interface --------------------
st.title("Hedge Fund Model Prediction Dashboard")
st.markdown("Enter the company information to predict if it's making a profit or a loss.")

# User input for stock name
stock_name = st.text_input("Enter the company stock name (e.g., TCS, RELIANCE):").strip().upper()

# User input for net income
net_income = st.number_input("Enter the Net Income (e.g., 123456.78):", min_value=-10000000.0, step=1000.0)

# Check if both inputs are provided
if stock_name and net_income is not None:
    try:
        # Encode the stock name using the saved label encoder
        stock_code = label_encoder.transform([stock_name])[0]
    except ValueError:
        st.error(f"❌ Error: '{stock_name}' was not found in the training data.")
        stock_code = None

    if stock_code is not None:
        # Prepare input data for prediction
        input_data = {
            'stock': stock_code,
            'netIncome_x': net_income,
        }

        # Fill missing features with default values (e.g., 0 for 'cash' and 'inventory')
        for feature in features:
            if feature not in input_data:
                input_data[feature] = 0.0  # Default value for missing features

        # Ensure the order of features matches the trained model
        input_array = np.array([[input_data[feature] for feature in features]])

        # Predict using the model
        prediction = model.predict(input_array)

        # Display result
        if prediction[0] == 1:
            st.success(f"✅ Prediction: The company '{stock_name}' is making a **PROFIT**.")
        else:
            st.error(f"❌ Prediction: The company '{stock_name}' is making a **LOSS**.")
