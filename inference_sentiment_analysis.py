import torch
import re
import numpy as np
import fasttext
from torch import nn
import os

# Step 1: Define the Sentiment Model Class
class SentimentRNN(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=2):
        super(SentimentRNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.rnn(x)
        out = self.fc(hn[-1])
        return out

MODEL_DIR = 'models'
fasttext_model_path = os.path.join(MODEL_DIR, "imdb_fasttext_supervised.bin")
pytorch_model_path = os.path.join(MODEL_DIR, 'imdb_sentiment_model.pth')

# Step 2: Load Models
ft_model = fasttext.load_model(fasttext_model_path)
model = SentimentRNN()
model.load_state_dict(torch.load(pytorch_model_path, weights_only=True))
model.eval()

# Step 3: Preprocess Input Text
def preprocess_text(text, embedding_model, max_len=100):
    review = re.sub(r'\W+', ' ', text.lower()).split()
    review_embed = np.zeros((max_len, 100))
    for i, word in enumerate(review[:max_len]):
        review_embed[i] = embedding_model.get_word_vector(word)
    return torch.tensor(review_embed, dtype=torch.float32).unsqueeze(0)

# Step 4: Predict Sentiment
def predict_sentiment(text):
    input_tensor = preprocess_text(text, ft_model)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    sentiment = "Positive" if predicted.item() == 1 else "Negative"
    print(f"Review: {text}\nPredicted Sentiment: {sentiment}")

# Example Usage
# sample_text = input("Enter a movie review: ")
# predict_sentiment(sample_text)

intersteller_movie = """Watching Interstellar was nothing short of an extraordinary journey that left me in awe long after the credits rolled. Directed by Christopher Nolan, this film transcends the boundaries of conventional cinema, taking us on a profound voyage through space and time that is as visually stunning as it is intellectually stimulating.

From the moment Interstellar begins, you are thrust into a future Earth on the brink of ecological collapse. The desperation and urgency are palpable, setting the stage for an epic quest to save humanity. What struck me immediately was the film's ability to blend grandiose science fiction with deeply personal storytelling. The relationship between Cooper, played masterfully by Matthew McConaughey, and his daughter Murph is the emotional heart of the film, grounding the cosmic adventure in a relatable human experience.

The visuals in Interstellar are nothing short of breathtaking. Nolan and his team have created a universe that feels both vast and intimate, filled with awe-inspiring imagery that ranges from the serene beauty of distant planets to the haunting isolation of space. The scenes involving the wormhole and black hole, in particular, are mesmerizing and showcase some of the best visual effects I’ve ever seen. These moments are not just eye candy; they are integral to the story, making the science of the film accessible and thrilling.

Hans Zimmer’s score is another standout element that elevates Interstellar to a whole new level. The music is powerful and haunting, perfectly complementing the film's themes of exploration and sacrifice. Zimmer’s use of the organ adds a unique, almost spiritual dimension to the soundtrack, making certain scenes resonate even more deeply.

What makes Interstellar truly exceptional is its ambitious exploration of complex scientific concepts. Nolan doesn’t shy away from diving into theories of relativity, time dilation, and the potential of other dimensions. Yet, he presents these ideas in a way that is engaging and understandable, even for those who might not be well-versed in astrophysics. The collaboration with physicist Kip Thorne ensures that the science is as accurate as it is fascinating, adding a layer of authenticity to the film.

The performances are stellar across the board. McConaughey delivers one of his finest performances, bringing depth and vulnerability to Cooper’s character. Anne Hathaway, Jessica Chastain, and Michael Caine also shine, each contributing to the film’s emotional and narrative complexity. Their performances ensure that, amidst the vastness of space, the story remains intimately human.

Interstellar is more than just a sci-fi epic; it’s a meditation on love, sacrifice, and the enduring human spirit. It challenges you to think about our place in the universe and the lengths we will go to protect those we love. The film’s ending, which I won’t spoil here, is both thought-provoking and deeply moving, leaving you with plenty to ponder.

Interstellar is a cinematic triumph that combines stunning visuals, a powerful score, and an emotionally charged story to create an unforgettable experience. It’s a film that speaks to both the heart and the mind, offering a journey that is as much about the exploration of space as it is about the exploration of human connections. Whether you’re a fan of science fiction or simply love great storytelling, Interstellar is a must-watch that will leave you inspired and in awe of the possibilities that lie beyond our world."""

negative_review = """movie is disguiting"""

predict_sentiment(intersteller_movie)
