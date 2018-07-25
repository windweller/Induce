"""
All the util functions related to data
such as building up a vocabulary, randomly initialize embedding etc.
"""

import torch
from torchtext import data, datasets


def init_emb(vocab, init="randn", num_special_toks=2, silent=False):
    # we can try randn or glorot
    # mode="unk"|"all", all means initialize everything
    emb_vectors = vocab.vectors
    sweep_range = len(vocab)
    running_norm = 0.
    num_non_zero = 0
    total_words = 0
    for i in range(num_special_toks, sweep_range):
        if len(emb_vectors[i, :].nonzero()) == 0:
            # std = 0.5 is based on the norm of average GloVE word vectors
            if init == "randn":
                torch.nn.init.normal(emb_vectors[i], mean=0, std=0.5)
        else:
            num_non_zero += 1
            running_norm += torch.norm(emb_vectors[i])
        total_words += 1
    if not silent:
        print("average GloVE norm is {}, number of known words are {}, total number of words are {}".format(
            running_norm / num_non_zero, num_non_zero, total_words))  # directly printing into Jupyter Notebook


class Dataset(object):
    def __init__(self):
        self.is_vocab_bulit = False
        self.iterators = []

        # self.TEXT, self.LABEL = None, None
        # self.train, self.val, self.test = None, None, None

    def build_vocab(self, config, silent=False):
        if config.emb_corpus == 'common_crawl':
            self.TEXT.build_vocab(self.train, vectors="glove.840B.300d")
            config.emb_dim = 300  # change the config emb dimension
        else:
            self.TEXT.build_vocab(self.train, vectors="glove.6B.{}d".format(config.emb_dim))

        self.LABEL.build_vocab(self.train)

        self.is_vocab_bulit = True
        self.vocab = self.TEXT.vocab

        if config.rand_unk:
            if not silent:
                print("initializing random vocabulary")
            init_emb(self.vocab, silent=silent)

    def get_iterators(self, device, train_batch_size=32, val_batch_size=128):
        if not self.is_vocab_bulit:
            raise Exception("Vocabulary is not built yet, needs to call build_vocab()")

        if len(self.iterators) > 0:
            return self.iterators  # return stored iterator

        # only get them after knowing the device (inside trainer or evaluator)
        train_iter, val_iter, test_iter = data.Iterator.splits(
            (self.train, self.val, self.test), sort_key=lambda x: len(x.text),  # no global sort, but within-batch-sort
            batch_sizes=(train_batch_size, val_batch_size, val_batch_size), device=device,
            sort_within_batch=True, repeat=False)

        return train_iter, val_iter, test_iter

    def unpack_batch(self, b):
        raise NotImplementedError

    def get_loss_fn(self):
        # different dataset should have different loss
        raise NotImplementedError

    @property
    def multilabel(self):
        raise NotImplementedError

    @property
    def label_size(self):
        raise NotImplementedError


class SSTDataset(Dataset):
    def __init__(self, config, train_subtrees=False, fine_grained=False):
        super(SSTDataset, self).__init__()

        self.TEXT = data.Field(sequential=True, include_lengths=True, batch_first=config.batch_first)
        self.LABEL = data.Field(sequential=False, batch_first=config.batch_first)
        self.fine_grained= fine_grained
        self.train, self.val, self.test = datasets.SST.splits(self.TEXT, self.LABEL,
                                                              train_subtrees=train_subtrees,
                                                              fine_grained=fine_grained)

    def unpack_batch(self, b):
        return b.text, b.label

    def get_loss_fn(self, size_average=True, reduce=False):
        return torch.nn.CrossEntropyLoss(size_average=size_average,
                                         reduce=reduce)

    @property
    def multilabel(self):
        return False

    @property
    def label_size(self):
        return 3 if not self.fine_grained else 5


# follow SentEval fashion...
class SSTB_Dataset(Dataset):
    def __init__(self, config):
        super(SSTB_Dataset, self).__init__()

        self.TEXT = data.Field(sequential=True, include_lengths=True, batch_first=config.batch_first)
        self.LABEL = data.Field(sequential=False, batch_first=config.batch_first)

        # TODO: need to add download link and such...


class CSUDataset(Dataset):
    pass

class PPDataset(Dataset):
    pass