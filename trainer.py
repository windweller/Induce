"""
Serves as an interactive console?
For longer running experiment with automatic evaluation

Also here we have all different configurations for all parts

Learning to Explain: An Information-Theoretic Perspective
on Model Interpretation
https://arxiv.org/pdf/1802.07814.pdf
"""

from utils import dotdict, Config


class GradientBaseConfig(Config):
    def __init__(self, taylor=False, embedding_level=True,
                 hidden_level=False, **kwargs):
        # taylor: whether to use Taylor expansion or not
        assert embedding_level or hidden_level, "can only choose either hidden level or embedding level"
        super(GradientBaseConfig, self).__init__(taylor=taylor,
                                                 embedding_level=embedding_level,
                                                 hidden_level=hidden_level,
                                                 **kwargs)

class LSTMBaseConfig(Config):
    def __init__(self, emb_dim=100, hidden_size=512, depth=1, label_size=42, bidir=False,
                dropout=0.2, emb_update=True, clip_grad=5., seed=1234,
                rand_unk=True, run_name="default", emb_corpus="gigaword",
                max_pool=False,
                 **kwargs):
        # run_name: the folder for the trainer
        super(LSTMBaseConfig, self).__init__(emb_dim=emb_dim,
                                             hidden_size=hidden_size,
                                             depth=depth,
                                             label_size=label_size,
                                             bidir=bidir,
                                             dropout=dropout,
                                             emb_update=emb_update,
                                             clip_grad=clip_grad,
                                             seed=seed,
                                             rand_unk=rand_unk,
                                             run_name=run_name,
                                             emb_corpus=emb_corpus,
                                             max_pool=max_pool,
                                             **kwargs)