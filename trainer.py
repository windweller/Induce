"""
Serves as an interactive console?
For longer running experiment with automatic evaluation

Also here we have all different configurations for all parts

Learning to Explain: An Information-Theoretic Perspective
on Model Interpretation
https://arxiv.org/pdf/1802.07814.pdf
"""

from utils import dotdict, Config


# we include TMPLA type normalization schemes into this configuration
class GradientBaseConfig(Config):
    def __init__(self, taylor=False,
                 embedding_level=False,
                 hidden_level=False,
                 sn_config=ScoreNormConfig(),
                 **kwargs):
        # taylor: whether to use Taylor expansion or not
        # sn_config: ScoreNormConfig
        assert embedding_level or hidden_level, "can only choose either hidden level or embedding level"
        super(GradientBaseConfig, self).__init__(taylor=taylor,
                                                 embedding_level=embedding_level,
                                                 hidden_level=hidden_level,
                                                 sn_config=sn_config,
                                                 **kwargs)


# Check the TMPLA paper for reference
# MultiLabel Normalization should work with any interpretation kernel
# including additive decomposition, contextual decomposition, etc.
class ScoreNormConfig(Config):
    def __init__(self,
                 local_norm=True,  # along each vector (d dimension) (Murdoch's paper did this)
                 global_norm=False,  # along all vectors (T dimension) (time dimension)
                 max_contrib=False,  # only pick the largest along d dimension for each T element
                 directional=False,  # compute directionl
                 **kwargs):
        assert local_norm or global_norm or max_contrib or directional, "Can only pick one setting for GMPBaseConfig"
        super(ScoreNormConfig, self).__init__(local_norm=local_norm,
                                              global_norm=global_norm,
                                              max_contrib=max_contrib,
                                              directional=directional,
                                              **kwargs)


class LSTMBaseConfig(Config):
    def __init__(self, emb_dim=100, hidden_size=512, depth=1, label_size=2, bidir=False,
                 dropout=0.2, emb_update=True, clip_grad=5., seed=1234,
                 rand_unk=True, run_name="default", emb_corpus="gigaword",
                 max_pool=False, attention=False, batch_first=True,
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
                                             attention=attention,
                                             batch_first=batch_first,
                                             **kwargs)


# here we store a list of common settings
gmpla = GradientBaseConfig(hidden_level=True,
                           sn_config=ScoreNormConfig(
                               max_contrib=True
                           ), taylor=True)
taylor_emb = GradientBaseConfig(taylor=True, embedding_level=True)
