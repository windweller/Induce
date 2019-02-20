import random
UNK = 'UNK'
# This file contains the dataset in a useful way. We populate a list of
# Trees to train/test our Neural Nets such that each Tree contains any
# number of Node objects.

# The best way to get a feel for how these objects are used in the program is to drop pdb.set_trace() in a few places throughout the codebase
# to see how the trees are used.. look where loadtrees() is called etc..


class Node:  # a node in the tree
    def __init__(self, label, word=None):
        self.label = label
        self.word = word
        self.parent = None  # reference to parent
        self.left = None  # reference to left child
        self.right = None  # reference to right child
        # true if I am a leaf (could have probably derived this from if I have
        # a word)
        self.isLeaf = False
        # true if we have finished performing fowardprop on this node (note,
        # there are many ways to implement the recursion.. some might not
        # require this flag)


class Tree:

    def __init__(self, treeString, openChar='(', closeChar=')'):
        tokens = []
        self.open = '('
        self.close = ')'
        for toks in treeString.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)
        # get list of labels as obtained through a post-order traversal
        self.labels = get_labels(self.root)
        self.num_words = len(self.labels)

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        split = 2  # position after open and label
        countOpen = countClose = 0

        if tokens[split] == self.open:
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1

        # New node
        node = Node(int(tokens[1]))  # zero index labels

        node.parent = parent

        # leaf Node
        if countOpen == 0:
            node.word = ''.join(tokens[2:-1]).lower()  # lower case?
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[2:split], parent=node)
        node.right = self.parse(tokens[split:-1], parent=node)

        return node

    def get_words(self):
        leaves = getLeaves(self.root)
        words = [node.word for node in leaves]
        return words


def leftTraverse(node, nodeFn=None, args=None):
    """
    Recursive function traverses tree
    from left to right. 
    Calls nodeFn at each node
    """
    if node is None:
        return
    leftTraverse(node.left, nodeFn, args)
    leftTraverse(node.right, nodeFn, args)
    nodeFn(node, args)


def getLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return getLeaves(node.left) + getLeaves(node.right)


def get_labels(node):
    if node is None:
        return []
    return get_labels(node.left) + get_labels(node.right) + [node.label]


def clearFprop(node, words):
    node.fprop = False


def loadTrees(dataSet='train'):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    file = 'trees/%s.filter.txt' % dataSet
    print("Loading %s trees.." % dataSet)
    with open(file, 'r') as fid:
        trees = [Tree(l.lower()) for l in fid.readlines()]

    return trees

def loadACDTrees(data, dataSet='train'):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    file = '%s/%s.normlabel.txt' % (data, dataSet)
    print("Loading %s trees.." % dataSet)
    with open(file, 'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]
    return trees

def simplified_data(num_train, num_dev, num_test, data='acd_trees_128d'):
    if 'acd' in data:
        train, dev, test = loadACDTrees(data, 'train'), loadACDTrees(data, 'dev'), loadACDTrees(data, 'test')
    else:
        train, dev, test = loadTrees('train'), loadTrees('dev'), loadTrees('test')

    #filter extreme trees, binarize labels
    train = [t for t in train if t.root.label != 2]
    dev = [t for t in dev if t.root.label != 2]
    test = [t for t in test if t.root.label != 2]
    binarize_labels(train)
    binarize_labels(dev)
    binarize_labels(test)

    return train, dev, test


def binarize_labels(trees):
    def binarize_node(node, _):
        if node.label < 2:
            node.label = 0
        elif node.label > 2:
            node.label = 1
    for tree in trees:
        leftTraverse(tree.root, binarize_node, None)
        tree.labels = get_labels(tree.root)
