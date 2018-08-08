import sys
from nltk import Tree


def get_sentences(path):
    sents = []
    with open(path, 'r') as f:
        for line in f:
            sent = Tree.fromstring(line.strip()).leaves()
            sents.append(sent)
    return sents

def get_trees(path):
    trees = []
    with open(path, 'r') as f:
        for line in f:
            tree = Tree.fromstring(line.strip().lower())
            trees.append(tree)
    return trees

# cut down origin dataset
def stage1(): 
    for filename in ['trees/train.txt', 'trees/dev.txt', 'trees/test.txt']:
        trees = get_sentences(filename)
        with open(filename) as fin:
            fout = open(filename.split('.')[0] + '.filter.txt', 'w')
            for i, line in enumerate(fin):
                if len(trees[i]) <= 20:
                    fout.write(line)
            fout.close()


# confirm each sentences are equal, do statistics
# result: all the same
def stage2():
    for filename in ['train', 'dev', 'test']:
        print filename
        acd_trees = get_sentences('acd_trees/%s.txt' % (filename))
        trees = get_sentences('trees/%s.filter.txt' % (filename))
        assert len(trees) == len(acd_trees)
        l = len(trees)
        for i in range(l):
            if ' '.join(trees[i]).lower() != ' '.join(acd_trees[i]):
                print ' '.join(trees[i]).lower() + '\n' + ' '.join(acd_trees[i])
                print len(' '.join(trees[i]).lower()), len('\n' + ' '.join(acd_trees[i]))


# check the differentce
def stage3():
    for filename in ['train', 'dev', 'test']:
        print filename
        acd_trees = get_trees('acd_trees/%s.normlabel.txt' % (filename))
        trees = get_trees('trees/%s.filter.txt' % (filename))
        l = len(trees)
        cnt, tot = 0, 0
        for i in range(l):
            tree_label = int(trees[i].label())
            acd_tree_label = float(acd_trees[i].label())
            if tree_label == 4:
                tot += 1
                if acd_tree_label > 0: cnt += 1
            elif tree_label == 0:
                tot += 1
                if acd_tree_label < 0: cnt += 1
            elif tree_label == 1 or tree_label == 2 or tree_label == 3:
                pass
            else:
                assert(0)
        print cnt, tot

# process data
def stage4():
    for filename in ['train', 'dev', 'test']:
        with open('acd_trees/%s.txt' % (filename)) as fin:
            fstd = open('trees/%s.filter.txt' % (filename))
            fout = open('acd_trees/%s.normlabel.txt' % (filename), 'w')
            for line in fin:
                newline = ''
                l = len(line)
                i = 0
                while i < l:
                    if line[i] == '(':
                        newline += line[i]
                        j = i + 5 if line[i + 5] == ' ' else i + 6
                        label = float(line[i + 1: j])
                        if label > 1.0: # <TODO> threshold
                            newline += '4'
                        elif label < -1.0:
                            newline += '0'
                        else:
                            newline += '2'
                        i = j
                    else:
                        newline += line[i]
                        i += 1
                stdline = fstd.readline()
                newline = newline[0] + stdline[1] + newline[2:]
                fout.write(newline)
            fout.close()
            fstd.close()

if __name__ == '__main__':
    # stage1()
    # stage2()
    stage3()
    # stage4()




