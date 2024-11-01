import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nltk.tokenize import word_tokenize
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Linear layers for query, key, and value
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        # Linear layer for output transformation
        self.out_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, mask=None):
        # Split input into multiple attention heads
        q = self.q_linear(q).view(-1, self.num_heads, self.head_dim)
        k = self.k_linear(k).view(-1, self.num_heads, self.head_dim)
        v = self.v_linear(v).view(-1, self.num_heads, self.head_dim)

        # Apply self-attention to each attention head
        attention_weights = torch.matmul(q, k.T) / math.sqrt(self.head_dim)
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -torch.inf)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        # Concatenate outputs of each attention head
        attention_output = attention_output.view(-1, self.hidden_size)

        # Apply output transformation
        output = self.out_linear(attention_output)
        return output


class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, max_length, vocab_size):
        super(Encoder, self).__init__()
        self.self_attention = nn.ModuleList([SelfAttention(hidden_size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.embedding_layer = EmbeddingLayer(vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(max_length, hidden_size)

    def forward(self, input_seq):
        # Convert the input sequence into list of tokens.
        input_tokens = word_tokenize(input_seq)

        # Pass the input tokens into an embedding layer.
        embedded_tokens = self.embedding_layer.forward(input_tokens)

        # Apply positional encoding
        encoded_tokens = self.positional_encoding(embedded_tokens)

        # todo: add the self attention to the encoder


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size):
        super(Decoder, self).__init__()
        self.self_attention = nn.ModuleList([SelfAttention(hidden_size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

    def forward(self, input_seq):
        # Implement the decoder logic here
        pass


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size=10_000, hidden_size=512):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_sequence):
        return self.embedding(input_sequence)


class PositionalEncoding(nn.Module):
    def __init__(self, max_length, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.pos_enc = self._get_pos_enc()

    def _get_pos_enc(self):
        pos_enc = np.zeros((self.max_length, self.embedding_dim))
        for pos in range(self.max_length):
            for i in range(self.embedding_dim):
                if i % 2 == 0:
                    pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.embedding_dim)))
                else:
                    pos_enc[pos, i] = np.cos(pos / (10000 ** ((2 * (i - 1)) / self.embedding_dim)))
        return torch.from_numpy(pos_enc).float()

    def forward(self, x):
        return x + self.pos_enc[:x.size(1), :]
