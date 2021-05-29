import gzip
import os

import wget
from tqdm import tqdm
from preprocess import Preprocessing
import pandas as pd

datasetUrl = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz'
dataPath = './data/reviews.json.gz'


def download_data():
    os.makedirs('data', exist_ok=True)

    if not os.path.exists(dataPath):
        print("File not accessible, downloading")
        wget.download(datasetUrl, out=dataPath)


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def get_df():
    i = 0
    df = {}
    for d in parse(dataPath):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def main():
    download_data()
    df = get_df()
    preprocessing = Preprocessing()

    for index in tqdm(range(len(df))):
        review = df.loc[index, 'reviewText']
        df.loc[index, 'reviewText'] = preprocessing.preprocess_text(review)

    df.to_csv("./data/dataset2.csv")


if __name__ == "__main__":
    main()