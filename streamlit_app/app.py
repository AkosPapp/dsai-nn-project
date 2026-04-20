import numpy as np
import pandas as pd
import streamlit as st
import onnxruntime as ort

MODEL_PATH = 'best_model.onnx'

# Update to match the NN input size from ml.ipynb
INPUT_SIZE = 200

st.set_page_config(page_title='NN Predictor', layout='wide')
st.title('NN Predictor')
st.caption('ONNX model loaded from best_model.onnx')

session = ort.InferenceSession(MODEL_PATH)

rows = (INPUT_SIZE + 3) // 4
columns = ['time', 'x', 'y', 'z']
if 'inputs_df' not in st.session_state:
    st.session_state['inputs_df'] = pd.DataFrame(0.0, index=range(rows), columns=columns)

st.subheader('Inputs')
edited_df = st.data_editor(
    st.session_state['inputs_df'],
    use_container_width=True,
    num_rows='fixed',
)

flat = edited_df[columns].to_numpy().reshape(-1)[:INPUT_SIZE]
input_array = np.array([flat], dtype=np.float32)
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_array})[0]

st.subheader('Prediction')
st.write(output.tolist()[0])
