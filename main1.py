import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

# -------------------------------
# Configuration and File Paths
# -------------------------------
MODEL_PATH = 'final_model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'
TRAINING_DATA_PATH = 'Sentences_75Agree_sample.txt'
MAX_WORDS = 5543  # Should match training configuration
VECTOR_MODE = 'binary'  # Ensure consistency with training

# -------------------------------
# Step 1. Load or Create the Tokenizer
# -------------------------------
if os.path.exists(TOKENIZER_PATH):
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer loaded from", TOKENIZER_PATH)
else:
    print("Tokenizer file not found. Creating a new one from training data.")
    
    if not os.path.exists(TRAINING_DATA_PATH):
        raise FileNotFoundError(f"Training data file not found at {TRAINING_DATA_PATH}")

    with open(TRAINING_DATA_PATH, 'r', encoding='latin1') as file:
        lines = file.readlines()

    data = [line.strip().split('@') for line in lines if "@" in line]
    df = pd.DataFrame(data, columns=['sentence', 'label'])

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['sentence'])

    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Tokenizer created and saved to", TOKENIZER_PATH)

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
CLASS_MAPPING = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}

def predict_sentiment(text):
    """
    Converts input text into a matrix using the tokenizer,
    obtains prediction probabilities from the model, and
    returns the sentiment label.
    """
    if not text.strip():
        return "Input text is empty. Please enter a valid sentence."

    vector = tokenizer.texts_to_matrix([text], mode=VECTOR_MODE)

    # Ensure the model input matches expected format
    vector = np.array(vector)

    predictions = model.predict(vector)
    print("Prediction probabilities:", predictions)

    predicted_class = np.argmax(predictions, axis=1)[0]
    return CLASS_MAPPING.get(predicted_class, "Unknown")

# -------------------------------
# Step 4. Main Program: Get User Input and Print the Result
# -------------------------------
if __name__ == "__main__":
    while True:
        user_text = input("Enter a sentence to analyze sentiment (or type 'exit' to quit): ").strip()
        if user_text.lower() == 'exit':
            print("Exiting program. Goodbye!")
            break
        sentiment_result = predict_sentiment(user_text)
        print(f"Sentiment: {sentiment_result}")
