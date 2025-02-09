import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

# Load dataset (used here to fit the tokenizer; adjust the filename if needed)
with open('Sentences_75Agree_sample.txt', 'r', encoding='latin1') as file:
    lines = file.readlines()

# Each line is assumed to be in the format: sentence@label
data = [line.strip().split('@') for line in lines]
df = pd.DataFrame(data, columns=['sentence', 'label'])

# You can optionally convert the labels if needed for training.
# For inference we only need the tokenizer and model.
# For example, if labels are strings and your training used a mapping, you might have:
# label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
# df['label'] = df['label'].map(label_mapping)

# Tokenizer configuration
# Ensure that num_words matches what was used during training.
max_words = 5543
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['sentence'])

# Load the pre-trained model
model = load_model('final_model.h5')

# Mapping of predicted class indices to sentiment labels
class_mapping = {
    0: 'Negative',
    1: 'Neutral',
    2: 'Positive'
}

def predict_sentiment(text):
    """
    Converts the input text into the representation expected by the model,
    makes a prediction, and returns the corresponding sentiment label.
    """
    # Convert the text to a vector using texts_to_matrix.
    # The 'mode' parameter should match what was used during training.
    vector = tokenizer.texts_to_matrix([text], mode='binary')
    
    # Get prediction probabilities (assuming the model outputs an array of shape (1,3))
    predictions = model.predict(vector)
    
    # Find the index with the highest probability
    class_index = np.argmax(predictions, axis=1)[0]
    
    # Map the index to a sentiment label
    sentiment = class_mapping.get(class_index, "Unknown")
    return sentiment

if __name__ == "__main__":
    # Get input from the user
    user_text = input("Enter a sentence to analyze sentiment: ")
    result = predict_sentiment(user_text)
    print(f"Sentiment: {result}")
