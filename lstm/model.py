"""
Contains:
LSTM (Bidir, attention / MaxPool)
Neural Rationale LSTM
Concrete Rationale LSTM
Murdoch's LSTM
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class LSTM(nn.Module):
    def __init__(self, model_config, vocab):
        super(LSTM, self).__init__()
        self.config = model_config
        self.attention = model_config.attention
        self.max_pool = model_config.max_pool
        self.batch_first = model_config.batch_first

        assert self.attention or self.max_pool, "Can only choose attention or max pooling"

        self.drop = nn.Dropout(model_config.dropout)  # embedding dropout
        self.encoder = nn.LSTM(
            model_config.emb_dim,
            model_config.hidden_size,
            model_config.depth,
            dropout=model_config.dropout,
            bidirectional=model_config.bidir,
            batch_first=self.batch_first)  # ha...not even bidirectional
        d_out = model_config.hidden_size if not model_config.bidir else model_config.hidden_size * 2
        self.out = nn.Linear(d_out, model_config.label_size)  # include bias, to prevent bias assignment

        self.embed = nn.Embedding(len(vocab), model_config.emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if model_config.emb_update else False

        self.trained = False

    def forward(self, input, lengths=None):
        output_vecs, hidden = self.get_vectors(input, lengths)
        return self.get_logits(output_vecs, hidden)

    def get_vectors(self, input, lengths=None):
        embed_input = self.embed(input)

        packed_emb = embed_input
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(embed_input, lengths, batch_first=self.batch_first)

        output, (hidden, c_n) = self.encoder(packed_emb)  # embed_input

        if lengths is not None:
            output = unpack(output, batch_first=self.batch_first)[0]

        # we ignored negative masking
        return output, hidden

    def get_logits(self, output_vec, hidden):
        if self.max_pool:
            pool_dim = 1 if self.batch_first else 0
            output = torch.max(output_vec, pool_dim)[0].squeeze(0)  # pool the temporal dim
        else:
            output = hidden

        return self.out(output)

    def get_softmax_weight(self):
        return self.out.weight
