import gzip
import os
import pandas as pd
import wget

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

