import streamlit as st
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

# Load TensorFlow Lite model
interpreter = Interpreter(model_path="perceptron_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Simple Perceptron Classifier")
st.write("Enter two binary values (0 or 1) to classify using TensorFlow Lite.")

# User inputs
x1 = st.number_input("Input X1 (0 or 1)", min_value=0, max_value=1, value=0)
x2 = st.number_input("Input X2 (0 or 1)", min_value=0, max_value=1, value=0)

if st.button("Predict"):
    input_data = np.array([[x1, x2]], dtype=np.float32)

    # Set input tensor and invoke the model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get the prediction output
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = 1 if prediction[0][0] > 0.5 else 0

    st.write(f"Predicted class: {predicted_class}")
