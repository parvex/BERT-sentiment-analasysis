import os

import torch
import sys

sys.path.insert(0, "..")

from transformers import BertTokenizer

from predict_review.predict_review import predict_single_review
from SentimentClassifier import SentimentClassifier
from consts import PRE_TRAINED_MODEL_NAME, class_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model = SentimentClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
    if not os.path.exists("model/model.pt"):
        os.makedirs("model", exist_ok=True)
        # TODO Download trained BERT from git

    model.state_dict(torch.load("model/model.pt"))
    print("BERT Sentiment Analyzer.")
    review = input("Please enter your review (or 'q' to exit):\n")
    while review != 'q':
        predict_single_review(review, tokenizer, model, device)
        print("_____________________________________________________")
        review = input("Please enter your review (or 'q' to exit):\n")

