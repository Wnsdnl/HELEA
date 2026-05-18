import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class BiEncoderModel(nn.Module):
    def __init__(self, model_name, init_scale=20.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls_embedding, p=2, dim=1)

    def save_pretrained(self, save_path):
        self.encoder.save_pretrained(save_path)
