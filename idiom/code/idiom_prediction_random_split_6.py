import json
import numpy as np
import random
random.seed(41)


def calculate_mean_p_h(r_file, match='Y'):
    with open(r_file, 'r') as f:
        all_prob = []
        all_hidden = []
        for i, line in enumerate(f):
            data = json.loads(line)
            if random.random() <= 0.5:  # random split
                data['match'] = 'Y'
            else:
                data['match'] = 'N'
            if data['match'] == match:
                prob = data['mean_prob']
                hidden = data['mean_hidden']

                all_prob.append(prob)
                all_hidden.append(hidden)
        all_prob = np.array(all_prob)
        all_hidden = np.array(all_hidden)
        mean_hidden = np.mean(all_hidden, axis=0)
        print("The statistics of probability of all match={} (number={}) is: mean={}, variance={}"
              .format(match, len(all_prob), all_prob.mean(), all_prob.var()))

        print("For hidden of all match={} (number={}) is: MEAN={}, VARIANCE={}"
              .format(match, len(all_hidden), mean_hidden.mean(), mean_hidden.var()))
        print('========')


def analyze_last_word_len(r_file, match='Y'):
    from collections import defaultdict
    len_count = defaultdict(int)
    with open(r_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
                lw_len = data['last_word_len']
                len_count[lw_len] += 1
    # increase order
    len_count = dict(sorted(len_count.items(), key=lambda item: item[0]))
    if match == "Y":
        print("The number of memorized is: {}".format(len_count))
    else:
        print("The number of NON-memorized is: {}".format(len_count))


def read_POS(r_file, match):
    from collections import defaultdict
    POS_count = defaultdict(int)
    with open(r_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
                idioms_pos = data['last_word_pos']
                POS_count[idioms_pos] += 1
    return POS_count


def analyze_pos(r_file):
    POS_Y_count = read_POS(r_file=r_file, match='Y')
    POS_N_count = read_POS(r_file=r_file, match='N')

    # POS_list = ["NUM", "AUX", "ADP", "ADV", "ADJ", "VERB",
    #             "NOUN", "PRON", "PROPN", "PART"]

    POS_list = ["ADV", "ADP", "VERB", "ADJ",
                "NOUN", "PRON", "PROPN"]

    for pos in POS_list:
        print("POS={}, Y_count={}, N_count={}, Y_ratio={}"
              .format(pos, POS_Y_count[pos], POS_N_count[pos],
                      POS_Y_count[pos] / (POS_Y_count[pos] + POS_N_count[pos])))


def analyze_idiom_count(r_file, match='Y'):
    from collections import defaultdict
    idiom_len_count = defaultdict(int)
    with open(r_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
                idiom_len = data['idiom_len']
                idiom_len_count[idiom_len] += 1
    # increase order
    len_count = dict(sorted(idiom_len_count.items(), key=lambda item: item[0]))
    if match == "Y":
        print("word number of idiom, #memorized={}".format(len_count))
    else:
        print("word number of idiom, #NON-memorized={}".format(len_count))


if __name__ == '__main__':
    # model = '7b'
    # model = '7b-chat'
    model = '13b'
    # model = '13b-chat'
    r_file = '../data/idiom_predict_{}.jsonl'.format(model)

    # analyze_idiom_count(r_file, match='Y')
    # analyze_idiom_count(r_file, match='N')
    # print('-----------------------')
    #
    # analyze_last_word_len(r_file=r_file, match='Y')
    # analyze_last_word_len(r_file=r_file, match='N')
    # print('-----------------------')
    #
    # analyze_pos(r_file=r_file)
    # print('-----------------------')

    # -----------------
    calculate_mean_p_h(r_file=r_file, match='Y')
    calculate_mean_p_h(r_file=r_file, match='N')
    print('-----------------------')





