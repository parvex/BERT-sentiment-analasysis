import torch
from torch.utils.data import Dataset


class ReviewDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, preprocessing, max_len: int):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.preprocessing = preprocessing
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        preprocessed_review = self.preprocessing.preprocess_text(review)
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            preprocessed_review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


