import json
import numpy as np


def calculate_mean_p_h(match='Y'):
    with open('idiom_predict.jsonl', 'r') as f:
        all_prob = []
        all_hidden = []
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
                prob = data['mean_prob']
                hidden = data['mean_hidden']

                all_prob.append(prob)
                all_hidden.append(hidden)
        all_prob = np.array(all_prob)
        all_hidden = np.array(all_hidden)
        mean_hidden = np.mean(all_hidden, axis=0)
        print("The mean probability of all match={} (number={}) is: {}".format(match, len(all_prob), all_prob.mean()))
        print("The variance probability of all match={} (number={}) is: {}".format(match, len(all_prob), all_prob.var()))

        print("The MEAN hidden of all match={} (number={}) is: {}".format(match, len(all_hidden), mean_hidden.mean()))
        print("The VARIANCE hidden of all match={} (number={}) is: {}".format(match, len(all_hidden), mean_hidden.var()))
        print('========')


def analyze_POS(match='Y'):
    from collections import defaultdict
    POS_count = defaultdict(int)
    with open('idiom_predict.jsonl', 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
                idioms_pos = data['last_word_pos']
                POS_count[idioms_pos] += 1
    # for pos, count in POS_count.items():
    #     print("match={}, POS={}, count={}".format(match, pos, count))
    return POS_count


if __name__ == '__main__':
    # calculate_mean_p_h(match='Y')
    # calculate_mean_p_h(match='N')

    POS_Y_count = analyze_POS(match='Y')
    POS_N_count = analyze_POS(match='N')

    # POS_list = ["NUM", "AUX", "ADP", "ADV", "ADJ", "VERB",
    #             "NOUN", "PRON", "PROPN", "PART"]
    POS_list = ["ADV", "ADP", "VERB", "ADJ",
                "NOUN", "PRON", "PROPN"]
    for pos in POS_list:
        print("POS={}, Y_count={}, N_count={}, Y_ratio={}".format(pos,
                                                                  POS_Y_count[pos],
                                                                  POS_N_count[pos],
                                                                  POS_Y_count[pos] / (POS_Y_count[pos] + POS_N_count[pos])))




