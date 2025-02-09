import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load dataset
with open('Sentences_75Agree_sample.txt', 'r', encoding='latin1') as file:
    lines = file.readlines()

data = [line.strip().split('@') for line in lines]
df = pd.DataFrame(data, columns=['sentence', 'label'])

df['label'] = df['label'].astype(int)

# Tokenization
max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['sentence'])
sequences = tokenizer.texts_to_sequences(df['sentence'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Load pre-trained model
model = load_model('final_model.h5')

# Make predictions
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prediction = model.predict(padded)
    return "Positive" if prediction > 0.5 else "Negative"

# Example usage
if __name__ == "__main__":
    sample_text = "This movie was fantastic!"
    print(f"Sentiment: {predict_sentiment(sample_text)}")
