import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# Load the model
model = load_model('chatbot_model.h5')


def preprocess_text(text):
    # Your preprocessing code here (tokenization, lemmatization, etc.)
    # For demonstration, this just returns the input text
    return text

def vectorize_text(text):
    # This should replicate the CountVectorizer transformation or similar used during training
    # For demonstration, this returns a numpy array of zeros of shape (1, input_shape)
    # Replace `input_shape` with the actual shape expected by your model
    input_shape = model.input_shape[1]  # Assuming model.input_shape is (None, input_shape)
    return np.zeros((1, input_shape))

# Streamlit app
def main():
    st.title("Chatbot Ecom")

    # User input
    user_input = st.text_area("Enter your text here", "")

    # Predict button
    if st.button("Submit your question"):
        # Preprocess and vectorize the user input
        preprocessed_input = preprocess_text(user_input)
        vectorized_input = vectorize_text(preprocessed_input)

        # Make a prediction
        prediction = model.predict(vectorized_input)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Assuming a categorical classification model

        # Display the prediction
        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Raw prediction output: {prediction}")

if __name__ == "__main__":
    main()
