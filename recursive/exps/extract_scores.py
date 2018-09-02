
def get_dev_test_root_score(file_name: str):
    dev_scores = []
    test_scores = []
    with open(dir_path + '/' + file_name, 'r') as f:
        state_dev = False
        state_test = False
        for line in f:
            if 'total_dev_loss' in line:
                state_dev = True
            if 'correct (root)' in line and state_dev:
                dev_scores.append(float(line.split("('correct (root) = ', ")[1].replace(')', '')))
                state_dev = False
            if 'total_test_loss' in line:
                state_test = True
            if 'correct (root)' in line and state_test:
                test_scores.append(float(line.split("('correct (root) = ', ")[1].replace(')', '')))
                state_test = False

    print('dev score: {} at epoch {}, test score: {}'.format(max(dev_scores), dev_scores.index(max(dev_scores)), test_scores[dev_scores.index(max(dev_scores))]))
    print('test score: {} at epoch {}'.format(max(test_scores), test_scores.index(max(test_scores))))

if __name__ == '__main__':
    # dir_path = '08-15-18-len20-sst-corpus'
    dir_path = '08-11-18-len20-sst-corpus'

    # get_dev_test_root_score('acd_128d_log_inter.txt')
    # get_dev_test_root_score('acd_512d_log_inter.txt')
    # get_dev_test_root_score('raw_log_inter.txt')
    # get_dev_test_root_score('acd_512d_log_no_inter.txt')

    # get_dev_test_root_score('raw_log_no_inter.txt')
    get_dev_test_root_score('acd_128d_log_no_inter.txt')