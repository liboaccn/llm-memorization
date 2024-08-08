import json
import numpy as np


def calculate_mean_p(r_file, match='Y'):
    with open(r_file, 'r') as f:
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
        print("The statistics of probability of all match={} (number={}) is: mean={}, variance={}"
              .format(match, len(all_prob), round(all_prob.mean(), 4), round(all_prob.var(), 4)))


def calculate_mean_h(r_file, match='Y'):
    with open(r_file, 'r') as f:
        all_mean = []
        all_var = []
        count = 0
        for i, line in enumerate(f):
            count = i+1
            data = json.loads(line)
            if data['match'] == match:
                # prob = data['mean_prob']
                hidden = np.array(data['mean_hidden'])
                mean = hidden.mean()
                var = hidden.var()

                all_mean.append(mean)
                all_var.append(var)

        all_mean = np.array(all_mean)
        all_var = np.array(all_var)

        _mean = np.mean(all_mean)
        _var = np.mean(all_var)

        # 保留小数点后4位
        _mean = round(_mean, 4)
        _var = round(_var, 4)

        print("Hidden: match={} (number={}，{}), MEAN={}, VARIANCE={}"
              .format(match, len(all_mean),len(all_mean)/count, _mean, _var))

    # with open(r_file, 'r') as f:
    #     # all_prob = []
    #     all_hidden = []
    #     for i, line in enumerate(f):
    #         data = json.loads(line)
    #         if data['match'] == match:
    #             prob = data['mean_prob']
    #             hidden = data['mean_hidden']
    #
    #             # all_prob.append(prob)
    #             all_hidden.append(hidden)
    #     # all_prob = np.array(all_prob)
    #     all_hidden = np.array(all_hidden)
    #     mean_hidden = np.mean(all_hidden, axis=0)
    #     # print("The statistics of probability of all match={} (number={}) is: mean={}, variance={}"
    #     #       .format(match, len(all_prob), all_prob.mean(), all_prob.var()))
    #
    #     print("For hidden of all match={} (number={}) is: MEAN={}, VARIANCE={}"
    #           .format(match, len(all_hidden), mean_hidden.mean(), mean_hidden.var()))


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
    new_len_count = defaultdict(int)
    for k, v in len_count.items():
        if k <=2:
            new_len_count[2] += v
        elif k <= 4:
            new_len_count[4] += v
        elif k <= 6:
            new_len_count[6] += v
        else:
            new_len_count[8] += v
    len_count = dict(sorted(new_len_count.items(), key=lambda item: item[0]))
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



def print_acc(r_file, match='Y'):
    total_count, acc_count = 0, 0
    all_prob = []
    with open(r_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            total_count += 1
            prob = data['mean_prob']
            all_prob.append(prob)
            if data['match'] == match:
                acc_count += 1
    print("The accuracy of match={} is: {}/{}={}".format(match, acc_count, total_count, round(acc_count/total_count, 3)))
    all_prob = np.array(all_prob)
    print("probability : mean={}"
          .format(round(all_prob.mean(), 3)))


if __name__ == '__main__':
    from load_LLMs import MODELS
    for model_name_or_path in MODELS:
        r_file = '../data/idiom_out_{}.jsonl'.format(model_name_or_path.split('/')[-1])
        print('=========== {} ==========='.format(r_file), '\n')

        print_acc(r_file, match='Y')

        # analyze_idiom_count(r_file, match='Y')
        # analyze_idiom_count(r_file, match='N')
        # print('-----------------------')

        # analyze_last_word_len(r_file=r_file, match='Y')
        # analyze_last_word_len(r_file=r_file, match='N')
        # print('-----------------------')

        # analyze_pos(r_file=r_file)
        # print('-----------------------')

        # calculate_mean_p(r_file=r_file, match='Y')
        # calculate_mean_p(r_file=r_file, match='N')

        # -----------------
        # calculate_mean_h(r_file=r_file, match='Y')
        # calculate_mean_h(r_file=r_file, match='N')
        print('-----------------------')





