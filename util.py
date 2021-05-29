import os
import sys

import requests

BERT_URL = "https://github.com/parvex/BERT-sentiment-analasysis/raw/BIG-BRANCH-WITH-BERT-USE-CAREFULLY/predict_review/model/model.pt"
BERT_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model", "model.pt")
DATASET_URL = "https://github.com/parvex/BERT-sentiment-analasysis/raw/BIG-BRANCH-WITH-BERT-USE-CAREFULLY/data/dataset.csv"
DATASET_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "dataset.csv")


def check_file_status(filepath, filesize):
    sys.stdout.write('\r')
    sys.stdout.flush()
    size = int(os.stat(filepath).st_size)
    percent_complete = (size/filesize)*100
    sys.stdout.write('%.3f %s' % (percent_complete, '% Completed'))
    sys.stdout.flush()


def download_file(file_url: str, target_path: str):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    req = requests.get(file_url, stream=True)
    filesize = int(req.headers['Content-length'])
    with open(target_path, 'wb') as outfile:
        chunk_size = 1048576
        for chunk in req.iter_content(chunk_size=chunk_size):
            outfile.write(chunk)
            if chunk_size < filesize:
                check_file_status(target_path, filesize)
        check_file_status(target_path, filesize)
        print()


def download_bert():
    print('Downloading BERT...')
    download_file(BERT_URL, BERT_path)


def download_dataset():
    print('Downloading dataset...')
    download_file(DATASET_URL, DATASET_PATH)