import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.out = nn.Linear(768, 1)
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.3)

    def forward(self, ids, mask, token_type_ids):
        out1 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        hidden_state = out1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        #bo = self.bert_drop(o2)
        output = self.out(pooler)
        return output
