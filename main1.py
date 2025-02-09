import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

# -----------------------------------------------------------------------------
# Step 1. Load dataset (used here to fit the tokenizer; adjust the filename if needed)
# -----------------------------------------------------------------------------
with open('Sentences_75Agree_sample.txt', 'r', encoding='latin1') as file:
    lines = file.readlines()

# Each line is assumed to be in the format: sentence@label
data = [line.strip().split('@') for line in lines]
df = pd.DataFrame(data, columns=['sentence', 'label'])

# If needed, convert labels to numeric using a mapping.
# (Uncomment and adjust if you want to convert labels from strings to numbers.)
# label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
# df['label'] = df['label'].map(label_mapping)

# -----------------------------------------------------------------------------
# Step 2. Set up the Tokenizer
# -----------------------------------------------------------------------------
# The model expects an input vector length matching max_words.
# Here, we set max_words to 5543, which should match your training configuration.
max_words = 5543
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['sentence'])

# -----------------------------------------------------------------------------
# Step 3. Load the Pre-Trained Model
# -----------------------------------------------------------------------------
model = load_model('final_model.h5')

# The following warnings are common:
#   - "Compiled the loaded model, but the compiled metrics have yet to be built..."
#   - "Error in loading the saved optimizer state..."
#
# These warnings do not affect inference.

# -----------------------------------------------------------------------------
# Step 4. Define Class Mapping and Prediction Function
# -----------------------------------------------------------------------------
# Ensure this mapping matches the order of outputs your model produces.
# For example, if the model outputs probabilities for [Negative, Neutral, Positive]:
class_mapping = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}

def predict_sentiment(text):
    """
    Converts the input text into a vector using texts_to_matrix,
    runs the model to predict sentiment, prints the raw prediction probabilities
    for debugging, and returns the sentiment label.
    """
    # Convert the input text to a fixed-length vector representation.
    # The 'mode' should match what was used during training (e.g., 'binary', 'tfidf', etc.)
    vector = tokenizer.texts_to_matrix([text], mode='binary')
    
    # Get prediction probabilities from the model.
    predictions = model.predict(vector)
    print("Prediction probabilities (debug):", predictions)
    
    # Get the index with the highest predicted probability.
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_sentiment = class_mapping.get(predicted_class, "Unknown")
    
    return predicted_sentiment

# -----------------------------------------------------------------------------
# Step 5. Main Program: Get User Input and Print the Result
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    user_text = input("Enter a sentence to analyze sentiment: ")
    sentiment_result = predict_sentiment(user_text)
    print(f"Sentiment: {sentiment_result}")
