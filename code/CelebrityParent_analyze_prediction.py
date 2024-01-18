import json
import numpy as np


def predict_parent_mean_p_h(match='Y', r_file=None):
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
        print("predict PARENT, The mean probability of all match={} (number={}) is: {}".format(match, len(all_prob), all_prob.mean()))
        print("predict PARENT, The variance probability of all match={} (number={}) is: {}".format(match, len(all_prob), all_prob.var()))

        print("predict PARENT, The MEAN HIDDEN of all match={} (number={}) is: {}".format(match, len(all_hidden), mean_hidden.mean()))
        print("predict PARENT, The VARIANCE HIDDEN of all match={} (number={}) is: {}".format(match, len(all_hidden), mean_hidden.var()))
        print('========')


def predict_child_mean_p_h(match='N', r_file=None):
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
        print("predict CHILD, The mean probability of all match={} (number={}) is: {}".format(match, len(all_prob), all_prob.mean()))
        print("predict CHILD, The variance probability of all match={} (number={}) is: {}".format(match, len(all_prob), all_prob.var()))

        print("predict CHILD, The MEAN HIDDEN of all match={} (number={}) is: {}".format(match, len(all_hidden), mean_hidden.mean()))
        print("predict CHILD, The VARIANCE HIDDEN of all match={} (number={}) is: {}".format(match, len(all_hidden), mean_hidden.var()))
        print('========')


if __name__ == '__main__':
    # predict_parent_mean_p_h(r_file='CelebrityParent_predict_parents_v1.json')
    # predict_child_mean_p_h(r_file='CelebrityParent_predict_child_v1.json')

    predict_parent_mean_p_h(r_file='CelebrityParent_predict_parents_v2.json')
    predict_child_mean_p_h(r_file='CelebrityParent_predict_child_v2.json')
