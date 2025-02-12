import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

# -------------------------------
# Configuration and File Paths
# -------------------------------
MODEL_PATH = 'final_model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'
TRAINING_DATA_PATH = 'Sentences_75Agree_sample.txt'
max_words = 5543  # Ensure this matches your training configuration
vectorization_mode = 'binary'  # Use the same mode as was used during training

# -------------------------------
# Step 1. Load or Create the Tokenizer
# -------------------------------
if os.path.exists(TOKENIZER_PATH):
    # Load the saved tokenizer
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer loaded from", TOKENIZER_PATH)
else:
    # Fallback: Fit a new tokenizer on the training data file
    print("Tokenizer pickle file not found. Fitting a new tokenizer from training data.")
    
    # Read the training data file (each line in format: sentence@label)
    with open(TRAINING_DATA_PATH, 'r', encoding='latin1') as file:
        lines = file.readlines()

    # Create a DataFrame from the data
    data = [line.strip().split('@') for line in lines if "@" in line]
    df = pd.DataFrame(data, columns=['sentence', 'label'])
    
    # Create and fit the tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['sentence'])
    
    # Save the tokenizer for future use
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Tokenizer fitted and saved to", TOKENIZER_PATH)

# -------------------------------
# Step 2. Load the Pre-Trained Model
# -------------------------------
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Model loaded from", MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

# -------------------------------
# Step 3. Define Class Mapping and Prediction Function
# -------------------------------
# Ensure that the mapping order matches your training output order.
# For example, if the model outputs probabilities for [Negative, Neutral, Positive]:
class_mapping = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}

def predict_sentiment(text):
    """
    Converts the input text into a vector using the tokenizer,
    obtains prediction probabilities from the model, prints them for debugging,
    and returns the sentiment label.
    """
    # Convert the input text to a fixed-length vector representation.
    vector = tokenizer.texts_to_matrix([text], mode=vectorization_mode)
    
    # Get prediction probabilities from the model.
    predictions = model.predict(vector)
    print("Prediction probabilities (debug):", predictions)
    
    # Determine the class with the highest probability.
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_sentiment = class_mapping.get(predicted_class, "Unknown")
    
    return predicted_sentiment

# -------------------------------
# Step 4. Main Program: Get User Input and Print the Result
# -------------------------------
if __name__ == "__main__":
    user_text = input("Enter a sentence to analyze sentiment: ")
    sentiment_result = predict_sentiment(user_text)
    print(f"Sentiment: {sentiment_result}")
