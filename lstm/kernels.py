"""
Interpretation kernels
"""

class GradientKernel(object):
    def __init__(self, config, model, loss_fn):
        self.config = config
        self.model = model
        self.loss_fn = loss_fn  # we assume this to be cross entropy loss

    def interpret(self, inputs, labels, lengths=None):
        self.model.zero_grad()

        logits = self.model(inputs, lengths)  # forward function of all models
        loss = self.loss_fn(logits, labels)

        # we backprop to get gradients
        loss.backward()

