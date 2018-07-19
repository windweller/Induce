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
    def __init__(self, vocab, config):
        super(LSTM, self).__init__()
        self.config = config
        self.drop = nn.Dropout(config.dropout)  # embedding dropout
        self.encoder = nn.LSTM(
            config.emb_dim,
            config.hidden_size,
            config.depth,
            dropout=config.dropout,
            bidirectional=config.bidir)  # ha...not even bidirectional
        d_out = config.hidden_size if not config.bidir else config.hidden_size * 2
        self.out = nn.Linear(d_out, config.label_size)  # include bias, to prevent bias assignment
        self.embed = nn.Embedding(len(vocab), config.emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embed.weight.requires_grad = True if config.emb_update else False

    def forward(self, input, lengths=None):
        output_vecs = self.get_vectors(input, lengths)
        return self.get_logits(output_vecs)

    def get_vectors(self, input, lengths=None):
        embed_input = self.embed(input)

        packed_emb = embed_input
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            packed_emb = nn.utils.rnn.pack_padded_sequence(embed_input, lengths)

        output, hidden = self.encoder(packed_emb)  # embed_input

        if lengths is not None:
            output = unpack(output)[0]

        # we ignored negative masking
        return output

    def get_logits(self, output_vec):
        output = torch.max(output_vec, 0)[0].squeeze(0)
        return self.out(output)

    def get_softmax_weight(self):
        return self.out.weight
