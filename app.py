import streamlit as st
import tflite_runtime.interpreter as tflite
import numpy as np

model = tf.keras.models.load_model('https://github.com/markleis/Finals/blob/main/perceptron_model.h5')

st.title("Simple Perceptron Classifier")
st.write("Enter two binary values (0 or 1) to classify using Perceptron.")

x1 = st.number_input("Input X1 (0 or 1)", min_value=0, max_value=1, value=0)
x2 = st.number_input("Input X2 (0 or 1)", min_value=0, max_value=1, value=0)

if st.button("Predict"):
    input_data = np.array([[x1, x2]], dtype=np.float32)
    prediction = model.predict(input_data)
    predicted_class = 1 if prediction > 0.5 else 0
    st.write(f"Predicted class: {predicted_class}")
