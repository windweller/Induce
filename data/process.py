for filename in ['dev.normlabel.txt', 'test.normlabel.txt', 'train.normlabel.txt']:
    with open(filename) as fin:
        out = []
        for line in fin:
            processed = line.strip().replace('0 ', '').replace('(', '( ').replace(')', ' )')
            tokens = processed.split(' ')
            correct_tokens = []
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i] == '(') and (tokens[i + 1] not in ['(', ')']):
                    correct_tokens.append(tokens[i + 1])
                    i += 3
                else:
                    correct_tokens.append(tokens[i])
                    i += 1
            correct_tokens.append(tokens[len(tokens) - 1])
            out.append(' '.join(correct_tokens))
        with open(filename + '.out', 'w') as fout:
            for item in out:
                fout.write(item + '\n')
