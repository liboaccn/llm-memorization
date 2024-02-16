import matplotlib.pyplot as plt
import seaborn as sns


def compute_cosine_similarity(array_a, array_b):
    from sklearn.metrics.pairwise import cosine_similarity

    similarity_list = []

    for i in range(array_a.shape[0]):
        sim = cosine_similarity(array_a[i].reshape(1, -1), array_b[i].reshape(1, -1))[0]
        similarity_list.extend(sim)

    return similarity_list


def plot_slope_graph(context_child, generated_parent, context_parent):
    # Assuming compute_cosine_similarity() is defined elsewhere and returns a cosine similarity score
    cosine_sim_1 = compute_cosine_similarity(generated_parent, context_child)
    cosine_sim_2 = compute_cosine_similarity(context_parent, context_child)

    # Setting up the figure and axes for the plot
    plt.figure(figsize=(6, 4))
    sns.set_style("whitegrid")

    # Defining the x and y coordinates for the two points
    x = ['Generated Parent', 'Context Parent']
    y = [cosine_sim_1, cosine_sim_2]

    # Plotting the slope graph
    plt.plot(x, y, marker='o', linestyle='-', color='b')

    # Adding title and labels for clarity
    plt.title('Cosine Similarity Transition')
    plt.ylabel('Cosine Similarity')

    # Showing the plot
    plt.show()

# Make sure to replace context_child, generated_parent, and context_parent with your actual data
# plot_slope_graph(context_child, generated_parent, context_parent)
