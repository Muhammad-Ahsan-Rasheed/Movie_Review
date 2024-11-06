# **Sentiment Analysis Model**

A PyTorch-based sentiment analysis project that classifies movie reviews as positive or negative using FastText embeddings and an LSTM-based neural network. The project includes a Streamlit application for interactive predictions.

## Working App
![image](https://github.com/user-attachments/assets/4ece67f7-2ab4-407d-9cbc-087a55e9dfa2)


## **Table of Contents**
- Project Overview
- File Structure
- Setup Instructions
- Running the Project
  - 1. Setting up the Environment
  - 2. Running the Inference Script
  - 3. Running the Streamlit App

## **Project Overview**

This project provides a sentiment classification model using:
- **FastText**: For generating word embeddings from the input text.
- **PyTorch**: For implementing an LSTM-based Recurrent Neural Network (RNN) that classifies reviews as positive or negative.
- **Streamlit**: For deploying an interactive web application where users can input text or select from sample reviews to test the model.

The model was trained on a movie review dataset and provides accurate sentiment classification.

---

## **File Structure**

```
.
├── data/                           # Directory for storing datasets
│   ├── imdb_data.csv               # Movie review dataset (if applicable)
│   └── additional_data.csv         # Additional datasets (if applicable)
├── models/                         # Directory for storing model files
│   ├── imdb_fasttext_supervised.bin  # FastText model for embeddings
│   └── imdb_sentiment_model.pth     # Trained PyTorch model for sentiment classification
├── plots/                          # Directory for storing plots (e.g., training curves)
│   └── training_plot.png           # Training and validation loss/accuracy plot
├── train_sentiment_analysis.py     # Script to train and save the sentiment analysis model
├── inference_sentiment_analysis.py # Script to run inference with the trained model
├── app.py                          # Streamlit application for interactive UI
├── requirements.txt                # List of dependencies required for the project
└── README.md                       # Project documentation
```

This structure includes:
- `data/` for datasets.
- `models/` for model files.
- `plots/` for saving plots like training curves.

## **Setup Instructions**

### 1. **Clone the Repository**
First, clone the repository from GitHub:

```bash
git clone https://github.com/Muhammad-Ahsan-Rasheed/Movie_Review.git
cd Movie_Review
```

### 2. **Create and Activate a Virtual Environment**
It is recommended to create a virtual environment to manage dependencies. You can create one with `venv`.

#### Using `venv`:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. **Install Dependencies**
Once the virtual environment is activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## **Running the Project**

### 1. **Setting up the Environment and Training the Model**
Ensure that all required datasets and directories are in place by running the training script. This will create the necessary directories and download the dataset if it's not already available.

```bash
python train_sentiment_analysis.py
```

The training script will:
- Download the dataset if not present.
- Train the sentiment analysis model.
- Save the trained model to the `models/` directory.
- Save the plots to the  `plots` directory.

### 2. **Running the Inference Script**
Once the model is trained, you can use the inference script to make predictions on movie reviews.

Run the `inference_sentiment_analysis.py` script to test the model with custom reviews:

```bash
python inference_sentiment_analysis.py
```

You can modify the script to include a review of your choice or add more functionality.

### 3. **Running the Streamlit App**
If you prefer an interactive UI to test the sentiment model, you can run the Streamlit app. This will allow you to enter your review or select from sample reviews to see the predicted sentiment (Positive or Negative).

```bash
streamlit run app.py
```

- **User Input**: You can either type your own review or click on buttons to use sample reviews.
- **Prediction**: The app will display the predicted sentiment for your input review.
