import os
import re
import urllib.request
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import fasttext
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# resource: Fasttext: https://hussainwali.medium.com/revolutionize-your-nlp-projects-with-fasttext-the-ultimate-guide-to-creating-and-using-word-7b8308513b50

# Set paths for data, models, and plots
DATA_DIR = 'data'
MODEL_DIR = 'models'
PLOT_DIR = 'plots'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

class SentimentRNN(nn.Module):
        def __init__(self, input_dim=100, hidden_dim=128, output_dim=2):
            super(SentimentRNN, self).__init__()
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            _, (hn, _) = self.rnn(x)
            out = self.fc(hn[-1])
            return out
        
# PyTorch Dataset class for IMDb data
class IMDbDataset(Dataset):
    def __init__(self, reviews, labels, embedding_model, max_len=100):
        self.reviews = reviews
        self.labels = labels
        self.embedding_model = embedding_model
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = re.sub(r'\W+', ' ', self.reviews[idx].lower()).split()
        label = self.labels[idx]

        # Embed words with FastText, pad or truncate to max_len
        review_embed = np.zeros((self.max_len, 100))
        for i, word in enumerate(review[:self.max_len]):
            review_embed[i] = self.embedding_model.get_word_vector(word)

        return torch.tensor(review_embed, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Function to download and extract the IMDb dataset
def download_and_extract_imdb(url, filename, dataset_folder):
    if not os.path.exists(filename):
        print("Downloading IMDb dataset...")
        urllib.request.urlretrieve(url, filename)

    if not os.path.exists(dataset_folder):
        print("Extracting IMDb dataset...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(DATA_DIR)

# Function to load reviews and labels from the IMDb dataset
def load_imdb_data(directory):
    reviews, labels = [], []
    for label_type in ['pos', 'neg']:
        path = os.path.join(directory, label_type)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                reviews.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    return reviews, labels

# Function to prepare data for FastText
def prepare_fasttext_data(reviews, labels, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for review, label in zip(reviews, labels):
            label_text = '__label__1' if label == 1 else '__label__0'
            f.write(f"{label_text} {re.sub(r'\\W+', ' ', review.lower())}\n")

# Function to train the FastText model
def train_fasttext_model(train_text_file, model_path):
    ft_model = fasttext.train_supervised(input=train_text_file, dim=100)
    ft_model.save_model(model_path)
    return ft_model


# Function to train the PyTorch model
def train_pytorch_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0
        for reviews, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            outputs = model(reviews)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        total_val_loss, correct_val, total_val = 0, 0, 0
        with torch.no_grad():
            for reviews, labels in val_loader:
                outputs = model(reviews)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Function to save the trained PyTorch model
def save_pytorch_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"PyTorch model saved at {model_path}")

# Function to plot and save training curves
def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, plot_dir, num_epochs):
    plt.figure(figsize=(10, 4))
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(num_epochs), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # Save plot image
    plot_path = os.path.join(plot_dir, 'training_validation_curves.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved at {plot_path}")

# Main function
def main():
    # Step 1: Download and extract the IMDb dataset
    imdb_url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    filename = os.path.join(DATA_DIR, 'aclImdb_v1.tar.gz')
    dataset_folder = os.path.join(DATA_DIR, 'aclImdb')
    download_and_extract_imdb(imdb_url, filename, dataset_folder)

    # Step 2: Load the IMDb dataset
    train_reviews, train_labels = load_imdb_data(os.path.join(dataset_folder, 'train'))

    # Step 3: Prepare data for FastText
    train_text_file = os.path.join(DATA_DIR, "imdb_train_labeled.txt")
    prepare_fasttext_data(train_reviews, train_labels, train_text_file)

    # Step 4: Train FastText model
    fasttext_model_path = os.path.join(MODEL_DIR, "imdb_fasttext_supervised.bin")
    ft_model = train_fasttext_model(train_text_file, fasttext_model_path)

    # Step 5: Create PyTorch Dataset and DataLoader
    full_dataset = IMDbDataset(train_reviews, train_labels, ft_model)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Step 6: Define PyTorch Model

    model = SentimentRNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 7: Train PyTorch Model
    num_epochs = 20
    train_losses, val_losses, train_accuracies, val_accuracies = train_pytorch_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    # Step 8: Save PyTorch Model
    pytorch_model_path = os.path.join(MODEL_DIR, "imdb_sentiment_model.pth")
    save_pytorch_model(model, pytorch_model_path)

    # Step 9: Plot and save training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, PLOT_DIR, num_epochs)

if __name__ == "__main__":
    main()
