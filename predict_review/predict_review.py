import torch

import sys
sys.path.insert(1, "..")

from consts import TOKEN_MAX_LEN, class_names
from preprocess import Preprocessing


def predict_single_review(review_text: str, preprocessing: Preprocessing, tokenizer, model, device):
    preprocessed_text = preprocessing.preprocess_text(review_text)
    encoded_review = tokenizer.encode_plus(
        preprocessed_text,
        max_length=TOKEN_MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        verbose=False
    )
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    print(f'Review text: {review_text}')
    print(f'Processed review text: {preprocessed_text}')
    print(f'Sentiment  : {class_names[prediction]}')
