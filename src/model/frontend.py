from transformers import Wav2Vec2Model
import torch.nn as nn
from torch import Tensor


class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
        self.out_dim = self.model.config.hidden_size

    def forward(self, input_data: Tensor) -> Tensor:
        self.model.eval()
        if input_data.ndim == 3:
            input_tmp = input_data.squeeze(-1)
        else:
            input_tmp = input_data
        emb = self.model(input_tmp).last_hidden_state
        return emb