import os

import requests
import torch
import sys

sys.path.insert(0, "..")

from transformers import BertTokenizer

from predict_review.predict_review import predict_single_review
from SentimentClassifier import SentimentClassifier
from consts import PRE_TRAINED_MODEL_NAME, class_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_URL = "https://github.com/parvex/BERT-sentiment-analasysis/raw/BIG-BRANCH-WITH-BERT-USE-CAREFULLY/predict_review/model/model.pt"

def check_file_status(filepath, filesize):
    sys.stdout.write('\r')
    sys.stdout.flush()
    size = int(os.stat(filepath).st_size)
    percent_complete = (size/filesize)*100
    sys.stdout.write('%.3f %s' % (percent_complete, '% Completed'))
    sys.stdout.flush()


def download_bert():
    os.makedirs("model", exist_ok=True)
    print('Downloading BERT...')
    req = requests.get(BERT_URL, stream=True)
    filesize = int(req.headers['Content-length'])
    with open(BERT_path, 'wb') as outfile:
        chunk_size = 1048576
        for chunk in req.iter_content(chunk_size=chunk_size):
            outfile.write(chunk)
            if chunk_size < filesize:
                check_file_status(BERT_path, filesize)
        check_file_status(BERT_path, filesize)
        print()


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model = SentimentClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
    model.to(device)
    BERT_path = "./model/model.pt"
    if not os.path.exists(BERT_path):
        download_bert()

    model.state_dict(torch.load(BERT_path))
    print("BERT Sentiment Analyzer.")
    review = input("Please enter your review (or 'q' to exit):\n")
    while review != 'q':
        predict_single_review(review, tokenizer, model, device)
        print("_____________________________________________________")
        review = input("Please enter your review (or 'q' to exit):\n")

