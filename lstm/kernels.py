"""
Interpretation kernels
"""

import torch

class GradientKernel(object):
    def __init__(self, config, model, loss_fn):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn  # we assume this to be cross entropy loss

        self.wrt_emb = config.embedding_level
        self.wrt_hidden = config.hidden_level
        self.taylor = config.taylor

    def compute_importance(self, score_matrix):
        # score_matrix: [batch_size, time_step, d]
        # d can be: embedding_size, or hidden_size
        # output: [batch_size, time_step]
        pass

    def interpret(self, inputs, labels, lengths=None):
        # GradientKernel requires all LSTM models to have `get_vectors()`
        # and `get_logits()`
        # model also need to have attribute `model.embed.weight`

        self.model.zero_grad()

        output_vecs, hidden = self.model.get_vectors(inputs, lengths)
        logits = self.model.get_logits(output_vecs, hidden)
        loss = self.loss_fn(logits, labels)

        # we compute gradients w.r.t. diff parameters
        if self.wrt_hidden:
            grads = torch.autograd.grad(loss, hidden)[0]
        elif self.wrt_emb:
            grads = torch.autograd.grad(loss, self.model.embed.weight)[0]

        # remember these grads have different shape

class AttentionKernel(object):
    def __init__(self, config, model, loss_fn):
        print("Attention Kernel requires modification to the model")