import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

# Load dataset
with open('Sentences_75Agree_sample.txt', 'r', encoding='latin1') as file:
    lines = file.readlines()

# Each line is split by '@' into sentence and label
data = [line.strip().split('@') for line in lines]
df = pd.DataFrame(data, columns=['sentence', 'label'])

# Convert label column to numeric values.
# Adjust the mapping according to your data.
label_mapping = {
    'negative': 0,
    'positive': 1
    # If you have additional labels (e.g., 'neutral'), add them here.
}
df['label'] = df['label'].map(label_mapping)

# Tokenization
# Set num_words to 5543 to match the expected input shape of the model.
max_words = 5543
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['sentence'])

# If needed for training, you might convert texts to a matrix:
# X = tokenizer.texts_to_matrix(df['sentence'], mode='binary')
# But for prediction we will convert a single text directly.

# Load pre-trained model
model = load_model('final_model.h5')

# Function to predict sentiment using texts_to_matrix
def predict_sentiment(text):
    # Convert the input text to a matrix representation.
    # You can change the mode ('binary', 'count', 'tfidf', or 'freq')
    # based on how your model was trained.
    vector = tokenizer.texts_to_matrix([text], mode='binary')
    prediction = model.predict(vector)
    # Assume the output is a probability in a 2D array (e.g., [[0.8]])
    prob = prediction[0][0]
    return "Positive" if prob > 0.5 else "Negative"

# Example usage
if __name__ == "__main__":
    sample_text = "This movie was fantastic!"
    print(f"Sentiment: {predict_sentiment(sample_text)}")
