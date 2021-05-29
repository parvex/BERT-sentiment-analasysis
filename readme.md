# Sentiment analysis with BERT on Amazon Reviews dataset

Dataset url: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz -o ./data/reviews.json.gz

How to use:
1. Setup conda environment with: ```conda env create -f environment.yml```
1. python -m spacy download en_core_web_sm
1. pip install -qq transformers
1. pip install wget
1. pip install pyenchant
1. apt-get install -y libenchant-dev
1. pip install spacy
1. Go to predict_review
1. Run python main.py and enjoy.

In learn.ipynb there's a training and analysis setup.