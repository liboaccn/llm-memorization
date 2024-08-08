import json
import numpy as np


def predict_parent_mean_p_h(match='Y', r_file=None):
    with open(r_file, 'r') as f:
        all_prob = []
        all_hidden = []
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
                prob = data['gen_parent_prob']
                hidden = data['gen_parent_hidden']

                all_prob.append(prob)
                all_hidden.append(hidden)
        all_prob = np.array(all_prob)
        all_hidden = np.array(all_hidden)
        mean_hidden = np.mean(all_hidden, axis=0)
        print("predict PARENT, The mean/variance probability of all match={} (number={}) is: mean={}, variance={}"
              .format(match, len(all_prob), all_prob.mean(), all_prob.var()))

        print("predict PARENT, The MEAN/VARIANCE HIDDEN of all match={} (number={}) is: MEAN={}, VARIANCE={}"
              .format(match, len(all_hidden), mean_hidden.mean(), mean_hidden.var()))
        print('========')


def predict_child_mean_p_h(match='N', r_file=None):
    with open(r_file, 'r') as f:
        all_prob = []
        all_hidden = []
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
                prob = data['gen_child_prob']
                hidden = data['gen_child_hidden']

                all_prob.append(prob)
                all_hidden.append(hidden)
        all_prob = np.array(all_prob)
        all_hidden = np.array(all_hidden)
        mean_hidden = np.mean(all_hidden, axis=0)
        print("predict CHILD, The mean/variance probability of all match={} (number={}) is: mean={}, variance={}"
              .format(match, len(all_prob), all_prob.mean(), all_prob.var()))

        print("predict CHILD, The MEAN/VARIANCE HIDDEN of all match={} (number={}) is: MEAN={}, VARIANCE={}"
              .format(match, len(all_hidden), mean_hidden.mean(), mean_hidden.var()))
        print('========')


def calculate_mean_p_h(r_file, match='Y'):
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
              .format(match, len(all_prob), all_prob.mean(), all_prob.var()))

        print("For hidden of all match={} (number={}) is: MEAN={}, VARIANCE={}"
              .format(match, len(all_hidden), mean_hidden.mean(), mean_hidden.var()))
        print('========')



if __name__ == '__main__':
    # model = "llama2-7b-hf"  # 7b, 13b
    # model = "llama2-13b-chat-hf"  # 7b, 13b

    from popQA.code.load_LLMs import MODELS
    for model_name_or_path in MODELS:
        r_file = '../data/LAMA_UHN_out_10_shot_{}.jsonl'.format(model_name_or_path.split('/')[-1])
        print('=========== {} ==========='.format(r_file), '\n')

        print('--------------------------------')

        calculate_mean_p_h(r_file=r_file, match='Y')
        calculate_mean_p_h(r_file=r_file, match='N')

        print('--------------------------------')