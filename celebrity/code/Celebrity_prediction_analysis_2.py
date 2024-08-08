import json
import numpy as np


def predict_parent_mean_p_h(match='Y', r_file=None, hidden_or_prob='prob'):
    with open(r_file, 'r') as f:
        all_prob = []
        all_hidden_mean = []
        all_hidden_var = []
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
                prob = data['gen_parent_prob']
                hidden = np.array(data['gen_parent_hidden'])

                all_prob.append(prob)
                all_hidden_mean.append(hidden.mean())
                all_hidden_var.append(hidden.var())
        all_prob = np.array(all_prob)
        _hidden_mean = round(np.array(all_hidden_mean).mean(), 4)
        _hidden_var = round(np.array(all_hidden_var).mean(), 4)

        if hidden_or_prob == 'prob':
            print("predict PARENT, The mean/variance probability of all match={} (number={}) is: mean={}, variance={}"
                  .format(match, len(all_prob), round(all_prob.mean(), 4), round(all_prob.var(), 4)))
        elif hidden_or_prob == 'hidden':
            print("predict PARENT, The MEAN/VARIANCE HIDDEN of all match={} (number={}) is: MEAN={}, VARIANCE={}"
                  .format(match, len(all_hidden_mean), _hidden_mean, _hidden_var))


def predict_child_mean_p_h(match='N', r_file=None, hidden_or_prob='prob'):
    with open(r_file, 'r') as f:
        all_prob = []
        all_hidden_mean = []
        all_hidden_var = []
        for i, line in enumerate(f):
            data = json.loads(line)
            if data['match'] == match:
                prob = data['gen_child_prob']
                hidden = np.array(data['gen_child_hidden'])

                all_prob.append(prob)
                all_hidden_mean.append(hidden.mean())
                all_hidden_var.append(hidden.var())
        all_prob = np.array(all_prob)
        all_hidden_mean = np.array(all_hidden_mean)
        all_hidden_var = np.array(all_hidden_var)

        _hidden_mean = round(all_hidden_mean.mean(), 4)
        _hidden_var = round(all_hidden_var.mean(), 4)

        if hidden_or_prob == 'prob':
            print("predict CHILD, The mean/variance probability of all match={} (number={}) is: mean={}, variance={}"
                  .format(match, len(all_prob), round(all_prob.mean(), 4), round(all_prob.var(), 4)))
        elif hidden_or_prob == 'hidden':
            print("predict CHILD, The MEAN/VARIANCE HIDDEN of all match={} (number={}) is: MEAN={}, VARIANCE={}"
                  .format(match, len(all_hidden_mean), _hidden_mean, _hidden_var))


def print_acc(r_file, match='Y'):
    total_count, acc_count = 0, 0
    with open(r_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            total_count += 1
            if data['match'] == match:
                acc_count += 1
    print("The accuracy of match={} is: {}/{}={}".format(match, acc_count, total_count, round(acc_count/total_count, 3)))


if __name__ == '__main__':
    from load_LLMs import MODELS
    # predict_parent_mean_p_h(r_file='CelebrityParent_predict_parents_{}_v1.json'.format(model))
    # predict_child_mean_p_h(r_file='CelebrityParent_predict_child_{}_v1.json'.format(model))
    # print('--------------------------------')

    # predict_parent_mean_p_h(r_file='CelebrityParent_predict_parents_{}_v3.json'.format(model))
    # predict_child_mean_p_h(r_file='CelebrityParent_predict_child_{}_v3.json'.format(model))

    # for model_name_or_path in MODELS:
    #     model_name = model_name_or_path.split('/')[-1]
    #     print('model_name:', model_name)
    #     predict_parent_mean_p_h(r_file='../data/celebrity_out_parents_{}.json'.format(model_name), hidden_or_prob='hidden')
    #     predict_child_mean_p_h(r_file='../data/celebrity_out_child_{}.json'.format(model_name), hidden_or_prob='hidden')
    #     print('--------------------------------')

    # for model_name_or_path in MODELS:
    #     model_name = model_name_or_path.split('/')[-1]
    #     print('model_name:', model_name)
    #     predict_parent_mean_p_h(r_file='../data/celebrity_out_parents_{}.json'.format(model_name), hidden_or_prob='prob')
    #     predict_child_mean_p_h(r_file='../data/celebrity_out_child_{}.json'.format(model_name), hidden_or_prob='prob')
    #     print('--------------------------------')

    for model_name_or_path in MODELS:
        model_name = model_name_or_path.split('/')[-1]
        print('model_name:', model_name)
        print_acc(r_file='../data/celebrity_out_parents_{}.json'.format(model_name), match='Y')
        print('--------------------------------')

