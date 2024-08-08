import json
import numpy as np


def read_file(r_file, matched='Y'):
    all_prob, all_acc = [], []
    with open(r_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)

            match = data['match']
            # if match != matched:
            #     continue
            prob = data['mean_prob']
            # prob = data['gen_parent_prob']
            acc = 1 if data['match'] == 'Y' else 0
            all_prob.append(prob)
            all_acc.append(acc)
    all_prob = np.array(all_prob)
    all_acc = np.array(all_acc)
    return all_prob, all_acc


def calculate_ece(probs, accuracy, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).

    Args:
        probs (numpy.ndarray): Array of predicted probabilities.
        acc (numpy.ndarray): classified or not.
        n_bins (int): Number of bins to divide the probability space.

    Returns:
        float: ECE score.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(probs)

    bin_errors = []

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        bin_mask = (probs >= bin_lower) & (probs < bin_upper)
        bin_probs = probs[bin_mask]
        bin_accuracy = accuracy[bin_mask]

        if len(bin_probs) > 0:
            bin_accuracy = np.mean(bin_accuracy)
            bin_confidence = np.mean(bin_probs)
            bin_size = len(bin_probs)
            bin_error = np.abs(bin_accuracy - bin_confidence)
            ece += bin_error * bin_size / total_samples
            bin_errors.append((bin_lower, bin_upper, bin_error))
        else:
            bin_errors.append((bin_lower, bin_upper, 0.0))

    return ece, bin_errors

if __name__ == '__main__':
    # r_file = "../data/idiom_out_llama2-13b.jsonl"
    # r_file = "D:\phd6\parttime\llm-memorization\poetry\data/shi_out_next_0_shot_qwen1_5-32b-chat.jsonl"
    # r_file = "D:\phd6\parttime\llm-memorization\proper\data/noun_out_llama2-13b.jsonl"
    # r_file = "D:\phd6\parttime\llm-memorization/terminology\data/term_out_llama2-13b.jsonl"
    # r_file = "D:\phd6\parttime\llm-memorization\celebrity\data/celebrity_out_parents_llama2-13b.json"
    # r_file = "D:\phd6\parttime\llm-memorization\popQA\data/popQA_out_10_shot_llama2-13b.jsonl"
    r_file = "D:\phd6\parttime\llm-memorization\LAMA_UHN\data/LAMA_UHN_out_10_shot_llama2-7b.jsonl"
    all_prob, all_acc = read_file(r_file, matched='Y')
    ece_score, bin_errors = calculate_ece(all_prob, all_acc, n_bins=2)

    print(f"ECE: {ece_score}")
    for bin_lower, bin_upper, bin_error in bin_errors:
        print(f"Bin [{bin_lower:.2f}, {bin_upper:.2f}): Calibration Error = {bin_error:.4f}")

