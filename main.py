import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load dataset
with open('Sentences_75Agree_sample.txt', 'r', encoding='latin1') as file:
    lines = file.readlines()

# Split each line by '@' into sentence and label
data = [line.strip().split('@') for line in lines]
df = pd.DataFrame(data, columns=['sentence', 'label'])

# Convert label column to numeric values.
# Option 1: Using a manual mapping. Adjust the mapping based on your data.
label_mapping = {
    'negative': 0,
    'positive': 1
    # If you have other labels (e.g., 'neutral'), include them here.
}
df['label'] = df['label'].map(label_mapping)

# Alternatively, if you simply want to convert to categorical codes automatically, you can use:
# df['label'] = df['label'].astype('category').cat.codes
# Note: The automatic codes are assigned in alphabetical order.

# Tokenization parameters
max_words = 10000
max_len = 100

# Create and fit the tokenizer on the sentences
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['sentence'])

# Convert sentences to sequences and pad them
sequences = tokenizer.texts_to_sequences(df['sentence'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Load the pre-trained model
model = load_model('final_model.h5')

# Function to predict sentiment of a given text
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prediction = model.predict(padded)
    # For binary classification, the prediction is typically a 2D array (e.g., [[0.8]])
    # We index the first element to get the probability.
    prob = prediction[0][0]
    return "Positive" if prob > 0.5 else "Negative"

# Example usage
if __name__ == "__main__":
    sample_text = "This movie was fantastic!"
    print(f"Sentiment: {predict_sentiment(sample_text)}")
