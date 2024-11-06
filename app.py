import streamlit as st
import torch
import re
import numpy as np
import fasttext
from torch import nn
import os

# Define the Sentiment Model Class
class SentimentRNN(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=2):
        super(SentimentRNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.rnn(x)
        out = self.fc(hn[-1])
        return out

# Load Models
MODEL_DIR = 'models'
fasttext_model_path = os.path.join(MODEL_DIR, "imdb_fasttext_supervised.bin")
pytorch_model_path = os.path.join(MODEL_DIR, 'imdb_sentiment_model.pth')

ft_model = fasttext.load_model(fasttext_model_path)
model = SentimentRNN()
model.load_state_dict(torch.load(pytorch_model_path, weights_only=True, ))
model.eval()

# Preprocess Text Function
def preprocess_text(text, embedding_model, max_len=100):
    review = re.sub(r'\W+', ' ', text.lower()).split()
    review_embed = np.zeros((max_len, 100))
    for i, word in enumerate(review[:max_len]):
        review_embed[i] = embedding_model.get_word_vector(word)
    return torch.tensor(review_embed, dtype=torch.float32).unsqueeze(0)

# Predict Sentiment Function
def predict_sentiment(text):
    input_tensor = preprocess_text(text, ft_model)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return "Positive" if predicted.item() == 1 else "Negative"

# Streamlit App Interface
st.title("Sentiment Analysis")
st.write("Enter a review or use one of the sample reviews to analyze its sentiment.")

# Sample reviews
sample_reviews = {
    "Positive Sample": """Watching Interstellar was nothing short of an extraordinary journey that left me in awe long after the credits rolled. Directed by Christopher Nolan, this film transcends the boundaries of conventional cinema, taking us on a profound voyage through space and time that is as visually stunning as it is intellectually stimulating.
        From the moment Interstellar begins, you are thrust into a future Earth on the brink of ecological collapse. The desperation and urgency are palpable, setting the stage for an epic quest to save humanity. What struck me immediately was the film's ability to blend grandiose science fiction with deeply personal storytelling. The relationship between Cooper, played masterfully by Matthew McConaughey, and his daughter Murph is the emotional heart of the film, grounding the cosmic adventure in a relatable human experience.
        The visuals in Interstellar are nothing short of breathtaking. Nolan and his team have created a universe that feels both vast and intimate, filled with awe-inspiring imagery that ranges from the serene beauty of distant planets to the haunting isolation of space. The scenes involving the wormhole and black hole, in particular, are mesmerizing and showcase some of the best visual effects I’ve ever seen. These moments are not just eye candy; they are integral to the story, making the science of the film accessible and thrilling.
        Hans Zimmer’s score is another standout element that elevates Interstellar to a whole new level.""",

    "Negative Sample": "The movie is disgusting and a waste of time.",
}

# User Input Section
user_input = st.text_area("Enter a review here:")

# Submit button for user input
submit_user_review = st.button("Submit Review")

# Sample Review Buttons
if st.button("Use Positive Sample"):
    review_text = sample_reviews["Positive Sample"]
    user_input = ""  # Clear user input
elif st.button("Use Negative Sample"):
    review_text = sample_reviews["Negative Sample"]
    user_input = ""  # Clear user input
elif submit_user_review and user_input:
    review_text = user_input  # Use user input if submitted
else:
    review_text = None  # No input to analyze

# Run Prediction if there's text to analyze
if review_text:
    sentiment = predict_sentiment(review_text)
    st.write(f"Predicted Sentiment: **{sentiment}**")
    st.write(f"Review: {review_text}")
