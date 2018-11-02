"""
Interpretation kernels

GradientKernel:
After computing a score for each label -- gradient kernel
figures out how each parameter contributes to each label

import numpy as np

B, T, d, L = 2, 3, 4, 5

x = np.random.rand(B, T, d, L)

i1 = np.repeat(np.arange(B), T)
i2 = np.tile(np.arange(T), B)
i3 = x.max(-1)[0].max(-1)[1].view(-1)

x_out = x[i1, i2, i3].view(B, T, L)

"""

import torch

class GradientKernel(object):
    def __init__(self, kernel_config, model):
        self.config = kernel_config
        self.model = model

        self.wrt_emb = kernel_config.embedding_level
        self.wrt_hidden = kernel_config.hidden_level
        self.taylor = kernel_config.taylor
        self.sn_config = kernel_config.sn_config

        self.softmax = torch.nn.Softmax()

    def normalize(self, score_matrix):
        # score_matrix: [batch_size, time_step, d, L]
        # d can be: embedding_size, or hidden_size
        # output: [batch_size, time_step]

        if self.sn_config.max_contrib and self.model.max_pool is False:
            print("max_contrib score normalization setting works best when using max pooling")

        # max_contrib and sum_contrib decides how to treat dimension d
        if self.sn_config.max_contrib:
            score_matrix = None  # most representative vote
        elif self.sn_config.sum_contrib:
            score_matrix = score_matrix.sum(dim=2)
        else:
            raise Exception("Must choose between max_contrib or sum_contrib")

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

    def interpret(self, inputs, lengths=None, return_raw=False):
        # GradientKernel requires all LSTM models to have `get_vectors()`
        # and `get_logits()`
        # model also need to have attribute `model.embed.weight`

        self.model.zero_grad()

        output_vecs, hidden = self.model.get_vectors(inputs, lengths)
        logits = self.model.get_logits(output_vecs, hidden)

        # we compute label logit w.r.t. diff parameters (how they are resopnsible for computing
        # such value)
        # then convert to scores
        scores = None
        if self.wrt_hidden:
            all_grads = []
            for label_i in range(self.model.label_size):
                grad = torch.autograd.grad(logits[label_i], hidden)[0]  # (B, T, d)
                score = grad * output_vecs if self.taylor else grad
                all_grads.append(score)
            # stack to (B, T, d, L)
            scores = torch.stack(all_grads, dim=-1)
        elif self.wrt_emb:
            all_grads = []
            for label_i in range(self.model.label_size):
                grad = torch.autograd.grad(logits[label_i], self.model.embed.weight)[0]
                score = grad * self.model.embed(inputs) if self.taylor else grad
                all_grads.append(score) # (B, T, w_d)
            # stack to (B, T, d, L)
            scores = torch.stack(all_grads, dim=-1)

        # then we have the score normalization scheme
        # compute importance
        return self.normalize(scores) if not return_raw else scores


class AttentionKernel(object):
    def __init__(self, model, loss_fn):
        if not model.attention:
            raise AttributeError("Attention Kernel requires modification to the model")


class NeuralRationaleKernel(object):
    def __init__(self, model, loss_fn):
        pass


class ConcreteRatinoaleKernel(object):
    def __init__(self, model, loss_fn):
        pass


ReinforceKernel = NeuralRationaleKernel
MutualInfoKernel = ConcreteRatinoaleKernel
