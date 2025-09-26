import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


st.set_page_config(
	title="Medical Insurance Cost Prediction",
	page_icon="💊",
	layout="centered",
)


@st.cache_resource
def load_model():
	model_path = Path("models/model.joblib")
	if not model_path.exists():
		return None
	try:
		return joblib.load(model_path)
	except Exception as e:
		st.error(f"Failed to load model from {model_path}: {e}")
		return None


model = load_model()

st.title("Medical Insurance Cost Prediction")
st.write("Enter patient details to estimate annual insurance charges.")

with st.sidebar:
	st.header("About")
	st.write(
		"This app predicts medical insurance charges using a machine learning model. "
		"Trained with a scikit-learn Pipeline that includes preprocessing."
	)
	st.markdown("**Repository files**: `app.py`, `train.py`, `models/model.joblib`")
	st.caption("Tip: Use the sample input below if unsure.")


def build_input_dataframe():
	col1, col2 = st.columns(2)
	with col1:
		age = st.number_input("Age", min_value=0, max_value=120, value=30)
		bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
		children = st.number_input("Children", min_value=0, max_value=10, value=0)
	with col2:
		sex = st.selectbox("Sex", ["male", "female"])
		smoker = st.selectbox("Smoker", ["yes", "no"])
		region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"]) 

	input_df = pd.DataFrame([
		{
			"age": age,
			"bmi": bmi,
			"children": children,
			"sex": sex,
			"smoker": smoker,
			"region": region,
		}
	])
	return input_df


def predict_charges(features_df: pd.DataFrame):
	if model is None:
		st.warning("Model file not found. Place your trained model at 'models/model.joblib'.")
		return None
	try:
		# If your exported object is a scikit-learn Pipeline including preprocessing,
		# you can pass raw features directly. Otherwise, insert preprocessing here.
		prediction = model.predict(features_df)
		return float(np.squeeze(prediction))
	except Exception as e:
		st.error(f"Prediction failed: {e}")
		return None


with st.form("prediction_form"):
	features = build_input_dataframe()
	with st.expander("Sample input", expanded=False):
		st.code(
			"""{
		  "age": 31,
		  "bmi": 27.5,
		  "children": 1,
		  "sex": "female",
		  "smoker": "no",
		  "region": "southeast"
		}""",
		)
	submitted = st.form_submit_button("Predict")

if submitted:
	pred = predict_charges(features)
	if pred is not None:
		st.success(f"Estimated annual charges: ${pred:,.2f}")


st.caption("Powered by Streamlit • Deploy on Streamlit Community Cloud")


