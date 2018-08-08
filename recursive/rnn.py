import sys
import os
import random
import numpy as np
import math
import time
import itertools
import shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import tree as tr
from utils import Vocab
from collections import OrderedDict
from random import shuffle


embed_size = 100
max_epochs = 30
lr = 0.01
l2 = 0.02
average_over = 700

class RNN_LSTM_Model(nn.Module):
  def __init__(self, vocab, embed_size=100, label_size=2):
    super(RNN_LSTM_Model, self).__init__()
    self.embed_size = embed_size
    self.label_size = label_size
    self.vocab = vocab
    self.embedding = nn.Embedding(int(self.vocab.total_words), self.embed_size)

    self.lstm = nn.LSTM(self.embed_size, self.embed_size, 1)
    self.projection = nn.Linear(self.embed_size, self.label_size , bias=True)

    self.node_list = []
    self.h = torch.zeros(1, 1, self.embed_size)
    self.c = torch.zeros(1, 1, self.embed_size)

  def init_variables(self):
    print("total_words = ", self.vocab.total_words)

  def walk_tree(self, in_node):
    if in_node.isLeaf:
      word_id = torch.LongTensor((self.vocab.encode(in_node.word), ))
      current_node = self.embedding(Variable(word_id).cuda())
      self.node_list.append(self.projection(current_node).unsqueeze(0))
    else:
      left  = self.walk_tree(in_node.left)
      right = self.walk_tree(in_node.right)
      x = torch.cat([left.unsqueeze(0), right.unsqueeze(0)], 0)
      output, (self.h, self.c) = self.lstm(x, (self.h, self.c))
      current_node = output[-1]  # we want time_len = -1
      self.node_list.append(self.projection(current_node).unsqueeze(0))
    return current_node

  def forward(self, x):
    """
    Forward function accepts input data and returns a Variable of output data
    """
    ### init lstm
    self.node_list = []
    self.h = torch.zeros(1, 1, self.embed_size).cuda()
    self.c = torch.zeros(1, 1, self.embed_size).cuda()
  
    root_node = self.walk_tree(x.root)
    all_nodes = torch.cat(self.node_list)
    #now I need to project out
    return all_nodes


class RNN_Model(nn.Module):
  def __init__(self, vocab, embed_size=100, label_size=2):
    super(RNN_Model, self).__init__()
    self.embed_size = embed_size
    self.label_size = label_size
    self.vocab = vocab
    self.embedding = nn.Embedding(int(self.vocab.total_words), self.embed_size)
    self.fcl = nn.Linear(self.embed_size, self.embed_size, bias=True)
    self.fcr = nn.Linear(self.embed_size, self.embed_size, bias=True)
    self.projection = nn.Linear(self.embed_size, self.label_size , bias=True)
    self.activation = F.relu
    self.node_list = []

  def init_variables(self):
    print("total_words = ", self.vocab.total_words)

  def walk_tree(self, in_node):
    if in_node.isLeaf:
      word_id = torch.LongTensor((self.vocab.encode(in_node.word), ))
      current_node = self.embedding(Variable(word_id).cuda())
      self.node_list.append(self.projection(current_node).unsqueeze(0))
    else:
      left  = self.walk_tree(in_node.left)
      right = self.walk_tree(in_node.right)
      current_node = self.activation(self.fcl(left) + self.fcl(right))
      self.node_list.append(self.projection(current_node).unsqueeze(0))
    return current_node

  def forward(self, x):
    """
    Forward function accepts input data and returns a Variable of output data
    """
    self.node_list = []
    root_node = self.walk_tree(x.root)
    all_nodes = torch.cat(self.node_list)
    #now I need to project out
    return all_nodes


if __name__ == '__main__':
  data = raw_input('Please input dataset: acd or raw\n')
  print data
  assert data == 'acd' or data == 'raw'
  train_data, dev_data, test_data = tr.simplified_data(0, 0, 0, data)
  print(len(train_data), len(dev_data), len(test_data))
  print(train_data[0])
  vocab = Vocab()
  train_sents = [t.get_words() for t in train_data]
  vocab.construct(list(itertools.chain.from_iterable(train_sents)))
  model = RNN_LSTM_Model(vocab, embed_size=embed_size).cuda()

  loss_history = []
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.0)

  for epoch in range(max_epochs):
    print("epoch = ", epoch)
    shuffle(train_data)
    total_root_prediction = 0.
    total_summed_accuracy = 0.
    if (epoch % 10 == 0) and epoch > 0:
        for param_group in optimizer.param_groups:
          #update learning rate
          print("Droping learning from %f to %f"%(param_group['lr'], 0.5 * param_group['lr']))
          param_group['lr'] = 0.5 * param_group['lr']
    for step, tree in enumerate(train_data):
      all_nodes = model(tree)

      labels  = []
      indices = []
      for x, y in enumerate(tree.labels):
        if y != 2:
          labels.append(y)
          indices.append(x)

      torch_labels = torch.LongTensor([l for l in labels if l != 2]).cuda()
      logits = all_nodes.index_select(dim=0, index=Variable(torch.LongTensor(indices)).cuda())
      logits_squeezed = logits.squeeze(1)
      predictions = logits.max(dim=2)[1].squeeze()

      correct = predictions.data == torch_labels
      #so correctly predicted (root);
      total_root_prediction += float(correct[-1])
      total_summed_accuracy += float(correct.sum()) / len(labels)

      objective_loss = F.cross_entropy(input=logits_squeezed, target=Variable(torch_labels))
      if objective_loss.data.item() > 5 and epoch > 10:
        #interested in phrase that have large loss (i.e. incorrectly classified)
        print(' '.join(tree.get_words()))

      loss_history.append(objective_loss.data.item())
      if step % 20 == 0 and step > 0:
        print("step %3d, last loss %0.3f, mean loss (%d steps) %0.3f" % (step, objective_loss.data.item(), average_over, np.mean(loss_history[-average_over:])))
      optimizer.zero_grad()

      if np.isnan(objective_loss.data.item()):
        print("object_loss was not a number")
        sys.exit(1)
      else:
        objective_loss.backward()
        clip_grad_norm_(model.parameters(), 5, norm_type=2.)
        optimizer.step()
    print("total root predicted correctly = ", total_root_prediction / float(len(train_data)))
    print("total node (including root) predicted correctly = ", total_summed_accuracy / float(len(train_data)))

    total_dev_loss = 0.
    dev_correct_at_root = 0.
    dev_correct_all = 0.
    for step, dev_example in enumerate(dev_data):
      all_nodes = model(dev_example)

      labels  = []
      indices = []
      for x, y in enumerate(dev_example.labels):
        if y != 2:
          labels.append(y)
          indices.append(x)
      torch_labels = torch.LongTensor([l for l in labels if l != 2]).cuda()
      logits = all_nodes.index_select(dim=0, index=Variable(torch.LongTensor(indices)).cuda())
      logits_squeezed = logits.squeeze(1)
      predictions = logits.max(dim=2)[1].squeeze()

      correct = predictions.data == torch_labels
      #so correctly predicted (root);
      dev_correct_at_root += float(correct[-1])
      dev_correct_all += float(correct.sum()) / len(labels)
      objective_loss = F.cross_entropy(input=logits_squeezed, target=Variable(torch_labels))
      total_dev_loss += objective_loss.data.item()
    print("total_dev_loss = ", total_dev_loss / float(len(dev_data)))
    print("correct (root) = ", dev_correct_at_root / float(len(dev_data)))
    print("correct (all)= ", dev_correct_all / float(len(dev_data)))

    total_test_loss = 0.
    test_correct_at_root = 0.
    test_correct_all = 0.
    for step, test_example in enumerate(test_data):
      all_nodes = model(test_example)

      labels  = []
      indices = []
      for x,y in enumerate(test_example.labels):
        if y != 2:
          labels.append(y)
          indices.append(x)
      torch_labels = torch.LongTensor([l for l in labels if l != 2]).cuda()
      logits = all_nodes.index_select(dim=0, index=Variable(torch.LongTensor(indices)).cuda())
      logits_squeezed = logits.squeeze(1)
      predictions = logits.max(dim=2)[1].squeeze()

      correct = predictions.data == torch_labels
      #so correctly predicted (root);
      test_correct_at_root += float(correct[-1])
      test_correct_all += float(correct.sum()) / len(labels)
      objective_loss = F.cross_entropy(input=logits_squeezed, target=Variable(torch_labels))
      total_test_loss += objective_loss.data.item()
    print("total_test_loss = ", total_test_loss / float(len(test_data)))
    print("correct (root) = ", test_correct_at_root / float(len(test_data)))
    print("correct (all)= ", test_correct_all / float(len(test_data)))
  print("DONE!")