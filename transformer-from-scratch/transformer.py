import time
import copy
from typing import Tuple
import datasets
from transformers import AutoTokenizer
import torch

from torch import Tensor

import math


class SelfAttentionHead(torch.nn.Module):
    def __init__(self, embedding_dim, query_dim, key_dim, value_dim):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.Wq = torch.nn.Linear(embedding_dim, query_dim, bias=False)
        self.Wk = torch.nn.Linear(embedding_dim, key_dim, bias=False)
        self.Wv = torch.nn.Linear(embedding_dim, value_dim, bias=False)

    def forward(self, x):
        q = torch.matmul(x, torch.transpose(self.Wq.weight.data, 0, 1))
        k = torch.matmul(x, torch.transpose(self.Wk.weight.data, 0, 1))
        v = torch.matmul(x, torch.transpose(self.Wv.weight.data, 0, 1))

        energy = torch.matmul(q, k.transpose(1, 2))
        normalized_energy = torch.softmax(energy / math.sqrt(self.key_dim), dim=2)
        out = torch.matmul(normalized_energy, v)

        return out


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, nheads, embedding_dim, query_dim, key_dim, value_dim):
        super().__init__()
        self.attention_heads = [
            SelfAttentionHead(embedding_dim, query_dim, key_dim, value_dim)
            for _ in range(nheads)
        ]
        self.Wo = torch.nn.Linear(nheads * value_dim, embedding_dim)

    def forward(self, x):
        output = torch.cat(tuple(ah(x) for ah in self.attention_heads), dim=2)
        output = self.Wo(output)
        return output


class EncoderLayer(torch.nn.Module):
    def __init__(self, nheads, embedding_dim, query_dim, key_dim, value_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(nheads, embedding_dim, query_dim, key_dim, value_dim)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)

        self.fully_connected = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim * 4, embedding_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim * 4, embedding_dim),
        )

        self.norm2 = torch.nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.multi_head_attention(x)
        x = self.norm1(x)
        x = x + self.fully_connected(x)
        x = self.norm2(x)

        return x


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Encoder(torch.nn.Module):
    def __init__(self, n_tokens, num_encoder_layers, nheads, embedding_dim, query_dim, key_dim, value_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_tokens, embedding_dim)
        self.encoder_layers = [
            EncoderLayer(nheads, embedding_dim, query_dim, key_dim, value_dim)
            for _ in range(num_encoder_layers)
        ]
        self.pe = PositionalEncoding(d_model=embedding_dim, max_len=embedding_dim // 2)
        self.classifier = torch.nn.Linear(embedding_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pe(x)
        for encoder in self.encoder_layers:
            x = encoder(x)
        x = self.classifier(x)
        x = torch.softmax(x, dim=1)
        return x


torch.manual_seed(42)
ah = Encoder(n_tokens=10000, num_encoder_layers=3, nheads=8, embedding_dim=256, query_dim=64, key_dim=64, value_dim=64)


tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')


data = datasets.load_dataset('imdb').with_format('torch')
train_data = data['train']
test_data = data['test']

ntokens = len(tok.get_vocab().keys())
model = Encoder(ntokens, num_encoder_layers=3, nheads=8,
                embedding_dim=256, query_dim=64, key_dim=64, value_dim=64)


criterion = torch.nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(model: torch.nn.Module, epoch: int) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    batch_size = 20
    start_time = time.time()

    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        tokenized = tok.batch_encode_plus(batch['text'], max_length=256,
                                          padding=True, return_tensors='pt')['input_ids']
        output = model(tokenized)
        loss = criterion(output.view(-1, ntokens), torch.tensor(batch['label']))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_data)/batch_size:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model: torch.nn.Module, eval_data: Tensor) -> float:
    pass


train(model, 1)

torch.save(model, '~/models/my-transformer')
