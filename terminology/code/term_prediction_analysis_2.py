import json
import numpy as np


def calculate_mean_p(r_file, match='Y'):
    with open(r_file, 'r', encoding='utf-8') as f:
        all_prob = []
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
                prob = data['mean_prob']
                all_prob.append(prob)
        all_prob = np.array(all_prob)
        print("The statistics of probability of all match={} (number={}) is: mean={}, variance={}"
              .format(match, len(all_prob), round(all_prob.mean(), 4), round(all_prob.var(), 4)))


def calculate_mean_h(r_file, match='Y'):
    with open(r_file, 'r', encoding='utf-8') as f:
        all_mean = []
        all_var = []
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
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

        print("Hidden: match={} (number={}), MEAN={}, VARIANCE={}"
              .format(match, len(all_mean), _mean, _var))


def analyze_last_word_len(r_file, match='Y'):
    from collections import defaultdict
    len_count = defaultdict(int)
    with open(r_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
                lw_len = data['answer_len']
                len_count[lw_len] += 1
    new_len_count = defaultdict(int)
    for k, v in len_count.items():
        if k <=5 :
            new_len_count[5] += v
        elif k <= 8:
            new_len_count[8] += v
        elif k <= 12:
            new_len_count[12] += v
        else:
            new_len_count[14] += v
    len_count = dict(sorted(new_len_count.items(), key=lambda item: item[0]))
    if match == "Y":
        print("The number of memorized is: {}".format(len_count))
    else:
        print("The number of NON-memorized is: {}".format(len_count))


def read_POS(r_file, match):
    from collections import defaultdict
    POS_count = defaultdict(int)
    with open(r_file, 'r', encoding='utf-8') as f:
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


def analyze_context_len(r_file, match='Y'):
    from collections import defaultdict
    term_len_count = defaultdict(int)
    with open(r_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
                term_count = data['term_count']
                term_len_count[term_count] += 1
    len_count = dict(sorted(term_len_count.items(), key=lambda item: item[0]))
    if match == "Y":
        print("term, context length, #memorized={}".format(len_count))
    else:
        print("term, context length, #NON-memorized={}".format(len_count))


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
    last_n = 1
    for model_name_or_path in MODELS:
        r_file =  '../data/term_out_{}.jsonl'.format(model_name_or_path.split('/')[-1])
        print('=========== {} ==========='.format(r_file), '\n')

        print_acc(r_file, match='Y')

        # analyze_context_len(r_file, match='Y')
        # analyze_context_len(r_file, match='N')
        # print('-----------------------')

        # analyze_last_word_len(r_file=r_file, match='Y')
        # analyze_last_word_len(r_file=r_file, match='N')
        # print('-----------------------')

        # analyze_pos(r_file=r_file)
        # print('-----------------------')

        # calculate_mean_p(r_file=r_file, match='Y')
        # calculate_mean_p(r_file=r_file, match='N')

        # calculate_mean_h(r_file=r_file, match='Y')
        # calculate_mean_h(r_file=r_file, match='N')
        # print('-----------------------')






