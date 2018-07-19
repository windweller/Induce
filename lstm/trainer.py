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

