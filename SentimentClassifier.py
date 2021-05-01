from torch import nn
from transformers import BertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes, pretrained_model_name):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)