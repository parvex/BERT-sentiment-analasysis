import os
import torch
import sys

from util import BERT_path, download_bert

sys.path.insert(0, "..")

from transformers import BertTokenizer

from preprocess import Preprocessing
from predict_review.predict_review import predict_single_review
from SentimentClassifier import SentimentClassifier
from consts import PRE_TRAINED_MODEL_NAME, class_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    preprocessing = Preprocessing()
    model = SentimentClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
    model.to(device)
    if not os.path.exists(BERT_path):
        download_bert()

    model.state_dict(torch.load(BERT_path))
    print("BERT Sentiment Analyzer.")
    review = input("Please enter your review (or 'q' to exit):\n")
    while review != 'q':
        predict_single_review(review, preprocessing, tokenizer, model, device)
        print("_____________________________________________________")
        review = input("Please enter your review (or 'q' to exit):\n")

