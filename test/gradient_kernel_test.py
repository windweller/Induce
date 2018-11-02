# https://jeffknupp.com/blog/2013/12/09/improve-your-python-understanding-unit-testing/

import unittest

from train import LSTMBaseConfig, GradientBaseConfig, Trainer
from lstm.model import LSTM
from lstm.kernels import GradientKernel
from data import SSTDataset

class GradientKernelTestCase(unittest.TestCase):
    """Tests for `primes.py`."""

    def test_max_pool_kernel(self):
        """Is five successfully determined to be prime?"""
        config = LSTMBaseConfig(max_pool=True)
        sst = SSTDataset(config)
        grad_config = GradientBaseConfig(hidden_level=True)

        lstm = LSTM(config, sst.vocab)

        grad_kernel = GradientKernel(grad_config, lstm, sst.get_loss_fn())

        trainer = Trainer(lstm, sst, config, device=0)

        trainer.train()

        grad_kernel.interpret()




if __name__ == '__main__':
    unittest.main()