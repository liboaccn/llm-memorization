import json
import os


os.environ['http_proxy'] = 'socks5://localhost:1080'
os.environ['https_proxy'] = 'socks5://localhost:1080'
os.environ["OPENAI_API_KEY"] = "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" #jiaang


def load_explainations(r_file):
    with open(r_file, 'r') as f:
        explainations = {}
        for i, line in enumerate(f):
            print(i, line)
            data = json.loads(line)
            idiom = data['idiom']
            exp = data['explanation']
            explainations[idiom] = exp
    return explainations


def dump_explainations(r_file, w_file, exps):
    with open(w_file, 'w') as fw:
        with open(r_file, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                idiom = data['idiom']
                print(i, idiom)
                explanation = exps[idiom]
                data['explanation'] = explanation
                json_data = json.dumps(data)
                fw.write(json_data + '\n')


def obtain_reps(r_file, w_file):
    from openai import OpenAI
    client = OpenAI()

    with open(w_file, 'w') as fw:
        with open(r_file, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                idiom = data['idiom']
                response = client.embeddings.create(
                    input=idiom,
                    model="text-embedding-3-large"
                )
                rep_i = response.data[0].embedding
                data['rep_i'] = rep_i

                explanation = data['explanation']
                response = client.embeddings.create(
                    input=explanation,
                    model="text-embedding-3-large"
                )
                rep_e = response.data[0].embedding
                data['rep_e'] = rep_e
                json_data = json.dumps(data)
                fw.write(json_data + '\n')
                fw.flush()


def tsne_visualization_idiom(r_file):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import json

    idiom_y = []
    idiom_n = []

    with open(r_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            idiom_v = data['rep_i']
            match = data['match']
            if match == 'Y':
                idiom_y.append(idiom_v)
            if match == 'N':
                idiom_n.append(idiom_v)
    group_idiom_y = np.array(idiom_y[:len(idiom_n)])
    group_idiom_n = np.array(idiom_n)
    combined_data = np.vstack((group_idiom_y, group_idiom_n))
    # Apply t-SNE
    tsne = TSNE(n_components=3, perplexity=100, learning_rate=50, n_iter=300, random_state=0,
                metric="euclidean", init="pca",)

    tsne_results = tsne.fit_transform(combined_data)

    # Split the results back into two groups
    tsne_a = tsne_results[:len(group_idiom_y)]
    tsne_b = tsne_results[len(group_idiom_n):]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_a[:, 0], tsne_a[:, 1], color='red', label='Memorized idiom')
    plt.scatter(tsne_b[:, 0], tsne_b[:, 1], color='blue', label='Non-memorized idiom')
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('idiom_rep.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def tsne_visualization_exp(r_file):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import json

    explanation_y = []
    explanation_n = []
    # Read in the data from a file
    with open(r_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            explanation_v = data['rep_e']
            match = data['match']
            if match == 'Y':
                explanation_y.append(explanation_v)
            if match == 'N':
                explanation_n.append(explanation_v)
    group_exp_y = np.array(explanation_y[:len(explanation_n)])
    group_exp_n = np.array(explanation_n)
    combined_data = np.vstack((group_exp_y, group_exp_n))
    # Apply t-SNE
    tsne = TSNE(n_components=3, perplexity=100, learning_rate=50, n_iter=300, random_state=0,
                metric="euclidean", init="pca",)

    tsne_results = tsne.fit_transform(combined_data)

    # Split the results back into two groups
    tsne_a = tsne_results[:len(group_exp_y)]
    tsne_b = tsne_results[len(group_exp_n):]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_a[:, 0], tsne_a[:, 1], color='red', label='Memorized explaination')
    plt.scatter(tsne_b[:, 0], tsne_b[:, 1], color='blue', label='Non-memorized explaination')
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('exp_rep.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def heatmap_idiom_exp(r_file):
    import numpy as np
    import json
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    import seaborn as sns

    idiom = []
    explanation = []
    number = 20
    count = 0
    with open(r_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            idiom_v = data['rep_i']
            explanation_v = data['rep_e']
            match = data['match']
            if match == 'N':
                idiom.append(idiom_v)
                explanation.append(explanation_v)
                count += 1
                if count >= number:
                    break
        count = 0
        for i, line in enumerate(f):
            data = json.loads(line)
            idiom_v = data['rep_i']
            explanation_v = data['rep_e']
            match = data['match']
            if match == 'Y':
                idiom.append(idiom_v)
                explanation.append(explanation_v)
                count += 1
                if count >= number:
                    break
    group_idiom = np.array(idiom)
    group_exp = np.array(explanation)

    cosine_sim = cosine_similarity(group_exp, group_idiom)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cosine_sim, annot=False, fmt=".2f", cmap='viridis', vmin=0.21, vmax=0.5)
    # plt.title('Cosine Similarity between Idiom Representations and Explanation Representations')
    plt.xlabel('Explanation representations', fontsize=14)
    plt.ylabel('Idiom representations', fontsize=14)
    plt.savefig('heatmap_idiom_exp.pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # step 1: explain idiom by gpt4, check its correctness by back translation
    # step 2: append the explaination key to idiom_predict_explaination.jsonl file
    # exps = load_explainations(r_file='idiom_explainations.jsonl')
    # r_file = "../code/idiom_predict.jsonl"
    w_file = "./idiom_predict_explaination.jsonl"
    # dump_explainations(r_file, w_file, exps)

    # step 3: obtain the representation of the idioms and explanations
    w2_file = './idiom_predict_explaination_rep.jsonl'
    # obtain_reps(r_file=w_file, w_file=w2_file)

    # step 4: visualize the representation
    # tsne_visualization_idiom(r_file=w2_file)
    # tsne_visualization_exp(r_file=w2_file)

    # step 5: obtain the representation of the idioms and explanations, heatmap
    heatmap_idiom_exp(r_file=w2_file)