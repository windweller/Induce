"""
Interpretation kernels
"""

import torch


class GradientKernel(object):
    def __init__(self, kernel_config, model, loss_fn):
        self.config = kernel_config
        self.model = model
        self.loss_fn = loss_fn  # we assume this to be cross entropy loss

        self.wrt_emb = kernel_config.embedding_level
        self.wrt_hidden = kernel_config.hidden_level
        self.taylor = kernel_config.taylor
        self.sn_config = kernel_config.sn_config

        self.softmax = torch.nn.Softmax()

    def normalize(self, score_matrix):
        # score_matrix: [batch_size, time_step, d]
        # d can be: embedding_size, or hidden_size
        # output: [batch_size, time_step]

        if self.sn_config.max_contrib and self.model.max_pool is False:
            print("max_contrib score normalization setting works best when computing max pooling")

        # 4 ways of normalization
        scores = None
        if self.sn_config.local_norm:  # normalize within each time step (on d dimension)
            # most common setting (used also by Murdoch)
            pos_scores = torch.clamp(score_matrix, min=0.)
            scores = pos_scores / pos_scores.sum(dim=2, keepdim=True)
        elif self.sn_config.global_norm:  # normalize importance w.r.t. a label (might not be very correct)
            pos_scores = torch.clamp(score_matrix, min=0.)
            scores = pos_scores / pos_scores.sum(dim=1, keepdim=True)
        elif self.sn_config.directional:
            scores = scores / torch.norm(scores, p=1, dim=1, keepdim=True)
            # this only makes sense w.r.t. a label (negative/positive contribution to a label)

        scores[scores != scores] = 0.  # make 'nan' = 0.

        return scores

    def interpret(self, inputs, labels, lengths=None, return_raw=False):
        # GradientKernel requires all LSTM models to have `get_vectors()`
        # and `get_logits()`
        # model also need to have attribute `model.embed.weight`

        self.model.zero_grad()

        output_vecs, hidden = self.model.get_vectors(inputs, lengths)
        logits = self.model.get_logits(output_vecs, hidden)
        loss = self.loss_fn(logits, labels)

        # we compute gradients w.r.t. diff parameters
        # then convert to scores
        scores = None
        if self.wrt_hidden:
            grads = torch.autograd.grad(loss, hidden)[0]
            # (B, T, d)
            scores = grads * output_vecs if self.taylor else grads
        elif self.wrt_emb:
            grads = torch.autograd.grad(loss, self.model.embed.weight)[0]
            scores = grads * self.model.embed(inputs) if self.taylor else grads
            # (B, T, w_d)

        # then we have the score normalization scheme
        # compute importance
        return self.normalize(scores) if not return_raw else scores


class AttentionKernel(object):
    def __init__(self, model, loss_fn):
        if not model.attention:
            raise AttributeError("Attention Kernel requires modification to the model")
