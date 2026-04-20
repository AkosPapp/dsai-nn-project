"""Streamlit dashboard for NN-based sensor correction.

Loads the ONNX model from MLflow Model Registry and provides:
- interactive input table for 50 samples (time, x, y, z)
- prediction output
- simple key-insights visualization of correction magnitudes
"""

import numpy as np
import pandas as pd
import streamlit as st
import onnxruntime as ort
import mlflow
import mlflow.onnx

MLFLOW_TRACKING_URI = 'http://localhost:5000/'
MODEL_URI = 'models:/BestNN_v1/Production'

# Update to match the NN input size from ml.ipynb
INPUT_SIZE = 200

st.set_page_config(page_title='NN Predictor', layout='wide')
st.title('NN Predictor')
st.caption('ONNX model loaded from MLflow Model Registry')

@st.cache_resource
def load_session():
    # Load ONNX model from MLflow registry and create an inference session.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    onnx_model = mlflow.onnx.load_model(MODEL_URI)
    return ort.InferenceSession(onnx_model.SerializeToString())

session = load_session()

# Build a fixed-size input table for 50 samples x 4 fields.
rows = (INPUT_SIZE + 3) // 4
columns = ['time', 'x', 'y', 'z']
if 'inputs_df' not in st.session_state:
    st.session_state['inputs_df'] = pd.DataFrame(0.0, index=range(rows), columns=columns)

st.subheader('Inputs')
edited_df = st.data_editor(
    st.session_state['inputs_df'],
    width='stretch',
    num_rows='fixed',
)

flat = edited_df[columns].to_numpy().reshape(-1)[:INPUT_SIZE]
input_array = np.array([flat], dtype=np.float32)
input_name = session.get_inputs()[0].name

if 'last_output' not in st.session_state:
    st.session_state['last_output'] = None

st.subheader('Prediction')
if st.button('Compute prediction') or st.session_state['last_output'] is None:
    st.session_state['last_output'] = session.run(None, {input_name: input_array})[0]

output = st.session_state['last_output']
if output is not None:
    pred = output.tolist()[0]
    st.write(pred)

    st.subheader('Key Insights')
    abs_pred = np.abs(np.array(pred, dtype=np.float32))
    insight_df = pd.DataFrame(
        {'axis': ['x', 'y', 'z'], 'abs_correction': abs_pred}
    ).set_index('axis')
    st.bar_chart(insight_df)
