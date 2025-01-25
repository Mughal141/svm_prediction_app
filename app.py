import streamlit as st
import joblib
import numpy as np

# Path to the pre-trained model
MODEL_PATH = "svm_model.pkl"

# Load the pre-trained SVM model
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        st.sidebar.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

# Function for making predictions
def make_prediction(model, input_data):
    try:
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Streamlit Application UI
st.title("SVM Prediction App")
st.header("Input Data for Prediction")

# Load the model
svm_model = load_model()

if svm_model:
    st.sidebar.header("Input Features")
    gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
    age = st.sidebar.number_input("Age", min_value=0.0, step=1.0)
    salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, step=1000.0)

    # Convert Gender to numerical values
    gender_encoded = 1 if gender == "Male" else 0
    user_input = [gender_encoded, age, salary]

    if st.sidebar.button("Predict"):
        prediction = make_prediction(svm_model, user_input)
        if prediction is not None:
            st.subheader(f"Prediction: {'Purchased' if prediction == 1 else 'Not Purchased'}")
else:
    st.error("Model could not be loaded. Please check the logs.")
