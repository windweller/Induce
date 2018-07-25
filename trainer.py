"""
Serves as an interactive console?
For longer running experiment with automatic evaluation

Also here we have all different configurations for all parts

Learning to Explain: An Information-Theoretic Perspective
on Model Interpretation
https://arxiv.org/pdf/1802.07814.pdf
"""

from utils import dotdict, Config, one_hot
import torch
import os
from os.path import join as pjoin
from torch import optim
import logging
import numpy as np
from sklearn import metrics

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


class LSTMBaseConfig(Config):
    def __init__(self, emb_dim=100, hidden_size=512, depth=1, label_size=2, bidir=False,
                 dropout=0.2, emb_update=True, clip_grad=5., seed=1234,
                 rand_unk=True, run_name="default", emb_corpus="gigaword",
                 max_pool=False, attention=False, batch_first=True,
                 log_interval=100,
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
                                             log_interval=log_interval,
                                             **kwargs)


# here we store a list of common settings
gmpla = GradientBaseConfig(hidden_level=True,
                           sn_config=ScoreNormConfig(
                               max_contrib=True
                           ), taylor=True)
taylor_emb = GradientBaseConfig(taylor=True, embedding_level=True)
max_pool_lstm = LSTMBaseConfig(max_pool=True)

# Typically Experiment manages trainers...
# but in this case we don't have Experimenters...yet...
class Trainer(object):
    def __init__(self, classifier, dataset, config, device, save_path='./sandbox', load=False, run_order=0):
        # a trainer loads model, or trains model, it's a wrapper around it
        # save_path: where to save log and model
        # run_order: randomized runs, or prepping for ensemble
        if load:
            # or we can add a new keyword...
            if os.path.exists(pjoin(save_path, 'model-{}.pickle'.format(run_order))):
                self.classifier = torch.load(pjoin(save_path, 'model-{}.pickle'.format(run_order))).cuda(device)
            else:
                self.classifier = torch.load(pjoin(save_path, 'model.pickle')).cuda(device)
        else:
            self.classifier = classifier.cuda(device)

        self.dataset = dataset
        self.device = device
        self.config = config
        self.save_path = save_path

        self.train_iter, self.val_iter, self.test_iter = self.dataset.get_iterators(device)
        self.external_test_iter = self.dataset.get_test_iterator(device)

        self.loss_fn = dataset.get_loss_fn()  # loss is specific to the dataset
        self.log_interval = config.log_interval

        need_grad = lambda x: x.requires_grad
        self.optimizer = optim.Adam(
            filter(need_grad, classifier.parameters()),
            lr=0.001)  # obviously we could use config to control this

        # setting up logging
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
        file_handler = logging.FileHandler("{0}/log.txt".format(save_path))
        self.logger = logging.getLogger(save_path.split('/')[-1])  # so that no model is sharing logger
        self.logger.addHandler(file_handler)

        self.logger.info(config)

    def load(self, run_order):
        self.classifier = torch.load(pjoin(self.save_path, 'model-{}.pickle').format(run_order)).cuda(self.device)

    def train(self, run_order=0, epochs=5, no_print=True):
        # train loop, as generic as possible!!
        exp_cost = None
        for e in range(epochs):
            self.classifier.train()
            for iter, data in enumerate(self.train_iter):
                self.classifier.zero_grad()
                (x, x_lengths), y = self.dataset.unpack_batch(data) # data.Text, data.Description

                logits = self.classifier(x, x_lengths)

                loss = self.loss_fn(logits, y).mean()
                loss.backward()

                torch.nn.utils.clip_grad_norm(self.classifier.parameters(), self.config.clip_grad)
                self.optimizer.step()

                if not exp_cost:
                    exp_cost = loss.data[0]
                else:
                    exp_cost = 0.99 * exp_cost + 0.01 * loss.data[0]

                if iter % self.log_interval == 0:
                    self.logger.info(
                        "iter {} lr={} train_loss={} exp_cost={} \n".format(iter, self.optimizer.param_groups[0]['lr'],
                                                                            loss.data[0], exp_cost))

            self.logger.info("enter validation...")
            valid_em, micro_tup, macro_tup = self.evaluate(is_test=False)
            self.logger.info("epoch {} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}\n".format(
                e + 1, self.optimizer.param_groups[0]['lr'], loss.data[0], valid_em
            ))

        # save model
        torch.save(self.classifier, pjoin(self.save_path, 'model-{}.pickle'.format(run_order)))

    def logit_to_preds(self, logits):
        # a unified way for both single-label and multi-label to generate prediction
        if self.dataset.multilabel:
            preds = (torch.sigmoid(logits) > 0.5).data.cpu().numpy().astype(float)
        else:
            preds = logits.data.max(1)[1]  # over label dim, and get indices
            preds = one_hot(preds, self.dataset.label_size).cpu().numpy()

        return preds

    def evaluate(self, is_test=False, silent=False, per_label=False, return_instances=False,
                 macro_non_zero=True):
        # is_test: True --> test set; False --> val set
        # returns: accuracy, (micro_p, m_r, m_f1), (macro_p, ma_r, ma_f1)

        # per_label: return (p, r, f1, s, accu), otherwise we return aggregate
        # return_instances: return a list of prediction and answers, useful for confusion matrix
        # macro_non_zero: compute only non-zero values

        self.classifier.eval()
        data_iter = self.test_iter if is_test else self.val_iter

        all_preds, all_y_labels = [], []

        # will have a evaluation function inside the dataset!
        for iter, data in enumerate(data_iter):
            (x, x_lengths), y = self.dataset.unpack_batch(data) # data.Text, data.Description
            logits = self.classifier(x, x_lengths)

            # preds = (torch.sigmoid(logits) > 0.5).data.cpu().numpy().astype(float)
            preds = self.logit_to_preds(logits)

            all_preds.append(preds)
            all_y_labels.append(y.data.cpu().numpy())

        preds = np.vstack(all_preds)
        ys = np.vstack(all_y_labels)

        if return_instances:
            return ys, preds

        if per_label:
            accu = np.array([metrics.accuracy_score(ys[:, i], preds[:, i]) for i in range(self.config.label_size)],
                            dtype='float32')
            p, r, f1, s = metrics.precision_recall_fscore_support(ys, preds, average=None)

            return p, r, f1, s, accu

        em = metrics.accuracy_score(ys, preds)  # normally accuracy, but becomes em in multilabel setting
        p, r, f1, s = metrics.precision_recall_fscore_support(ys, preds, average=None)
        micro_p, micro_r, micro_f1 = np.average(p, weights=s), np.average(r, weights=s), np.average(f1, weights=s)

        if macro_non_zero:
            macro_p, macro_r, macro_f1 = np.average(p[p.nonzero()]), np.average(r[r.nonzero()]), np.average(f1[f1.nonzero()])
        else:
            macro_p, macro_r, macro_f1 = np.average(p), np.average(r), np.average(f1)

        return em, (micro_p, micro_r, micro_f1), (macro_p, macro_r, macro_f1)